use std::collections::HashMap;
use nuts_rs::{CpuLogpFunc, CpuMathError, LogpError, Storable};
use nuts_storable::HasDims;
use thiserror::Error;
use faer::{Mat, Par, Spec};
use faer::linalg::cholesky::llt::{factor, solve, inverse};
use faer::linalg::cholesky::llt::factor::LltParams;
use faer::dyn_stack::{MemBuffer, MemStack};
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

pub struct GeneratedLogp {
    // Preallocated faer matrices (reused across calls, no allocation in hot loop)
    k_mat: Mat<f64>,      // N×N kernel matrix → becomes L after factorization
    alpha: Mat<f64>,      // N×1 for solve (K^{-1} y)
    kinv: Mat<f64>,       // N×N for inverse
    dk_dls: Vec<f64>,     // N*N derivative dK/d(log_ls) (flat for fast access)
    chol_buf: MemBuffer,  // scratch for cholesky_in_place
    inv_buf: MemBuffer,   // scratch for inverse
}

impl GeneratedLogp {
    pub fn new() -> Self {
        let chol_scratch = factor::cholesky_in_place_scratch::<f64>(
            N, Par::Seq, Spec::<LltParams, f64>::default(),
        );
        let inv_scratch = inverse::inverse_scratch::<f64>(N, Par::Seq);
        Self {
            k_mat: Mat::zeros(N, N),
            alpha: Mat::zeros(N, 1),
            kinv: Mat::zeros(N, N),
            dk_dls: vec![0.0; N * N],
            chol_buf: MemBuffer::new(chol_scratch),
            inv_buf: MemBuffer::new(inv_scratch),
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
        let dk_dls = &mut self.dk_dls;
        let k = &mut self.k_mat;
        for i in 0..N {
            for j in 0..=i {
                let d = X_DATA[i] - X_DATA[j];
                let d_sq = d * d;
                let r_sq_scaled = d_sq * inv_ls_sq;
                let exp_term = (-0.5 * r_sq_scaled).exp();
                let k_ij = eta_sq * exp_term;
                let dk_ij = k_ij * r_sq_scaled;

                if i == j {
                    k[(i, j)] = k_ij + sigma_sq + JITTER;
                } else {
                    k[(i, j)] = k_ij;
                    k[(j, i)] = k_ij;
                }
                dk_dls[i * N + j] = dk_ij;
                dk_dls[j * N + i] = dk_ij;
            }
        }

        // === Cholesky decomposition in-place via faer ===
        factor::cholesky_in_place(
            k.as_mut(),
            Default::default(),
            Par::Seq,
            MemStack::new(&mut self.chol_buf),
            Spec::<LltParams, f64>::default(),
        ).map_err(|_| {
            SampleError::Recoverable("Cholesky failed: not positive definite".to_string())
        })?;
        // Now k_mat lower triangle is L

        // === Log determinant: log|K| = 2 * sum(log(L_ii)) ===
        let mut log_det = 0.0;
        for i in 0..N {
            log_det += k[(i, i)].ln();
        }
        log_det *= 2.0;

        // === Solve K alpha = y in-place: alpha starts as y, becomes K^{-1} y ===
        let alpha = &mut self.alpha;
        for i in 0..N {
            alpha[(i, 0)] = Y_DATA[i];
        }
        solve::solve_in_place(
            k.as_ref(),
            alpha.as_mut(),
            Par::Seq,
            MemStack::new(&mut []),
        );

        // === GP log-likelihood ===
        let mut yt_kinv_y = 0.0;
        for i in 0..N {
            yt_kinv_y += Y_DATA[i] * alpha[(i, 0)];
        }
        logp += -0.5 * (N as f64 * LN_2PI + log_det + yt_kinv_y);

        // === Compute K^{-1} in-place via faer ===
        inverse::inverse(
            self.kinv.as_mut(),
            k.as_ref(),
            Par::Seq,
            MemStack::new(&mut self.inv_buf),
        );
        // Note: inverse only fills lower triangle, mirror to upper
        let kinv = &self.kinv;
        // faer's inverse for Cholesky fills lower triangle, need to symmetrize
        // Actually looking at the source: it does L_inv^H * L_inv which gives full symmetric result
        // via triangular matmul with TriangularLower output structure.
        // But the output is only lower triangle. Let's use both halves via symmetry.

        // === Gradients ===
        let mut grad_log_ls = 0.0;
        let mut grad_log_eta = 0.0;
        let mut grad_log_sigma = 0.0;

        for i in 0..N {
            let alpha_i = alpha[(i, 0)];
            for j in 0..N {
                let alpha_j = alpha[(j, 0)];
                // kinv is symmetric but only lower triangle stored
                let kinv_ij = if i >= j { kinv[(i, j)] } else { kinv[(j, i)] };
                let w_ij = kinv_ij - alpha_i * alpha_j;

                grad_log_ls += w_ij * dk_dls[i * N + j];

                let d = X_DATA[i] - X_DATA[j];
                let kernel_ij = eta_sq * (-0.5 * d * d * inv_ls_sq).exp();
                grad_log_eta += w_ij * 2.0 * kernel_ij;

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
