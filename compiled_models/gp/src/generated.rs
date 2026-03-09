use std::collections::HashMap;
use nuts_rs::{CpuLogpFunc, CpuMathError, LogpError, Storable};
use nuts_storable::HasDims;
use thiserror::Error;
use crate::data::*;

#[derive(Debug, Error)]
pub enum SampleError {
    #[error("Recoverable: {0}")]
    Recoverable(String),
}

impl LogpError for SampleError {
    fn is_recoverable(&self) -> bool { true }
}

/// GP regression: y ~ MvNormal(0, K)
/// K_ij = eta^2 * exp(-0.5 * (x_i - x_j)^2 / ls^2) + sigma^2 * delta_ij
///
/// Unconstrained parameters: [log_ls, log_eta, log_sigma]
/// Priors: ls ~ HalfNormal(5), eta ~ HalfNormal(5), sigma ~ HalfNormal(5)
pub const N_PARAMS: usize = 3;
const LN_2PI: f64 = 1.8378770664093453;
const JITTER: f64 = 1e-6;

#[derive(Storable, Clone)]
pub struct Draw {
    #[storable(dims("param"))]
    pub parameters: Vec<f64>,
}

#[derive(Clone)]
pub struct GeneratedLogp {
    // Preallocated workspace to avoid allocation in hot loop
    k: Vec<f64>,        // N*N kernel matrix (lower triangle after Cholesky)
    alpha: Vec<f64>,    // N-vector: K^{-1} y
    dk_dls: Vec<f64>,   // N*N derivative dK/d(log_ls)
    kinv: Vec<f64>,     // N*N: K^{-1}
    tmp: Vec<f64>,      // N-vector scratch
}

impl GeneratedLogp {
    pub fn new() -> Self {
        Self {
            k: vec![0.0; N * N],
            alpha: vec![0.0; N],
            dk_dls: vec![0.0; N * N],
            kinv: vec![0.0; N * N],
            tmp: vec![0.0; N],
        }
    }
}

impl HasDims for GeneratedLogp {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        HashMap::from([("param".to_string(), N_PARAMS as u64)])
    }
}

impl CpuLogpFunc for GeneratedLogp {
    type LogpError = SampleError;
    type FlowParameters = ();
    type ExpandedVector = Draw;

    fn dim(&self) -> usize { N_PARAMS }

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, SampleError> {
        let log_ls = position[0];
        let log_eta = position[1];
        let log_sigma = position[2];

        let ls = log_ls.exp();
        let eta = log_eta.exp();
        let sigma = log_sigma.exp();

        let ls_sq = ls * ls;
        let eta_sq = eta * eta;
        let sigma_sq = sigma * sigma;
        let inv_ls_sq = 1.0 / ls_sq;

        gradient[0] = 0.0;
        gradient[1] = 0.0;
        gradient[2] = 0.0;

        let mut logp = 0.0;

        // === Priors (HalfNormal(5) with LogTransform) ===
        // For each: logp = ln(2) - 0.5*ln(2π) - ln(5) - 0.5*(x/5)^2 + log_x
        let prior_const = 2.0f64.ln() - 0.5 * LN_2PI - 5.0f64.ln();

        let ls_s = ls * 0.2;
        logp += prior_const - 0.5 * ls_s * ls_s + log_ls;
        gradient[0] += -0.04 * ls_sq + 1.0;

        let eta_s = eta * 0.2;
        logp += prior_const - 0.5 * eta_s * eta_s + log_eta;
        gradient[1] += -0.04 * eta_sq + 1.0;

        let sigma_s = sigma * 0.2;
        logp += prior_const - 0.5 * sigma_s * sigma_s + log_sigma;
        gradient[2] += -0.04 * sigma_sq + 1.0;

        // === Build kernel matrix K and dK/d(log_ls) ===
        let k = &mut self.k;
        let dk_dls = &mut self.dk_dls;

        for i in 0..N {
            for j in 0..=i {
                let d = X_DATA[i] - X_DATA[j];
                let d_sq = d * d;
                let r_sq_scaled = d_sq * inv_ls_sq;
                let exp_term = (-0.5 * r_sq_scaled).exp();
                let k_ij = eta_sq * exp_term;

                // dK/d(log_ls) = eta^2 * exp(...) * d^2/ls^2  (chain rule: d/d(log_ls) = ls * d/d(ls))
                let dk_ij = k_ij * r_sq_scaled;

                if i == j {
                    k[i * N + j] = k_ij + sigma_sq + JITTER;
                } else {
                    k[i * N + j] = k_ij;
                    k[j * N + i] = k_ij;
                }
                dk_dls[i * N + j] = dk_ij;
                dk_dls[j * N + i] = dk_ij;
            }
        }

        // === Cholesky decomposition: K = L L^T ===
        // In-place: lower triangle of k becomes L
        for j in 0..N {
            let mut sum = k[j * N + j];
            for p in 0..j {
                sum -= k[j * N + p] * k[j * N + p];
            }
            if sum <= 0.0 {
                return Err(SampleError::Recoverable("Cholesky failed: not positive definite".to_string()));
            }
            let ljj = sum.sqrt();
            k[j * N + j] = ljj;
            let inv_ljj = 1.0 / ljj;

            for i in (j + 1)..N {
                let mut sum = k[i * N + j];
                for p in 0..j {
                    sum -= k[i * N + p] * k[j * N + p];
                }
                k[i * N + j] = sum * inv_ljj;
            }
        }
        // Now k[i*N+j] for i >= j is L[i,j]

        // === Log determinant: log|K| = 2 * sum(log(L_ii)) ===
        let mut log_det = 0.0;
        for i in 0..N {
            log_det += k[i * N + i].ln();
        }
        log_det *= 2.0;

        // === Solve L z = y (forward substitution) ===
        let alpha = &mut self.alpha;
        for i in 0..N {
            let mut sum = Y_DATA[i];
            for j in 0..i {
                sum -= k[i * N + j] * alpha[j];
            }
            alpha[i] = sum / k[i * N + i];
        }

        // === Solve L^T alpha = z (back substitution) ===
        for i in (0..N).rev() {
            let mut sum = alpha[i];
            for j in (i + 1)..N {
                sum -= k[j * N + i] * alpha[j];
            }
            alpha[i] = sum / k[i * N + i];
        }
        // Now alpha = K^{-1} y

        // === GP log-likelihood ===
        // logp = -0.5 * (N*ln(2π) + log|K| + y^T K^{-1} y)
        let mut yt_kinv_y = 0.0;
        for i in 0..N {
            yt_kinv_y += Y_DATA[i] * alpha[i];
        }

        logp += -0.5 * (N as f64 * LN_2PI + log_det + yt_kinv_y);

        // === Compute K^{-1} by solving L L^T X = I column by column ===
        let kinv = &mut self.kinv;
        let tmp = &mut self.tmp;

        for col in 0..N {
            // Forward solve: L z = e_col
            for i in 0..N {
                let rhs = if i == col { 1.0 } else { 0.0 };
                let mut sum = rhs;
                for j in 0..i {
                    sum -= k[i * N + j] * tmp[j];
                }
                tmp[i] = sum / k[i * N + i];
            }

            // Back solve: L^T x = z
            for i in (0..N).rev() {
                let mut sum = tmp[i];
                for j in (i + 1)..N {
                    sum -= k[j * N + i] * tmp[j];
                }
                tmp[i] = sum / k[i * N + i];
            }

            for i in 0..N {
                kinv[i * N + col] = tmp[i];
            }
        }

        // === Gradients ===
        // W = K^{-1} - alpha * alpha^T
        // dlogp/dθ = -0.5 * tr(W * dK/dθ)
        //
        // For log_eta: dK/d(log_eta) = 2 * eta^2 * exp(-0.5 * d^2/ls^2) = 2*(K - sigma^2*I - jitter*I) for off-diag
        //              For diagonal: 2 * eta^2 (the kernel part only)
        // For log_sigma: dK/d(log_sigma) = 2 * sigma^2 * I
        // For log_ls: precomputed dk_dls

        let mut grad_log_ls = 0.0;
        let mut grad_log_eta = 0.0;
        let mut grad_log_sigma = 0.0;

        for i in 0..N {
            for j in 0..N {
                let w_ij = kinv[i * N + j] - alpha[i] * alpha[j];

                // dK/d(log_ls)
                grad_log_ls += w_ij * dk_dls[i * N + j];

                // dK/d(log_eta): 2 * (kernel part of K_ij)
                // kernel_ij = K_ij - (sigma^2 + jitter) * delta_ij
                // But we need the original kernel values. Since dk_dls stores
                // eta^2 * exp(...) * d^2/ls^2, and kernel = eta^2 * exp(...),
                // we can reconstruct: for i!=j, dK/d(log_eta) = 2 * K_ij
                // for i==j, dK/d(log_eta) = 2 * (K_ii - sigma^2 - jitter)
                // But simpler: recompute inline
                let d = X_DATA[i] - X_DATA[j];
                let kernel_ij = eta_sq * (-0.5 * d * d * inv_ls_sq).exp();
                grad_log_eta += w_ij * 2.0 * kernel_ij;

                // dK/d(log_sigma): 2 * sigma^2 * delta_ij
                if i == j {
                    grad_log_sigma += w_ij * 2.0 * sigma_sq;
                }
            }
        }

        gradient[0] += -0.5 * grad_log_ls;
        gradient[1] += -0.5 * grad_log_eta;
        gradient[2] += -0.5 * grad_log_sigma;

        Ok(logp)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self, _rng: &mut R, array: &[f64],
    ) -> Result<Draw, CpuMathError> {
        Ok(Draw { parameters: array.to_vec() })
    }
}
