use std::collections::HashMap;
use std::simd::prelude::*;
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
    k: Vec<f64>,
    alpha: Vec<f64>,
    dk_dls: Vec<f64>,
    kinv: Vec<f64>,
    tmp: Vec<f64>,
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

        // Priors
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

        // Build kernel matrix - SIMD for inner loop of each row
        let k = &mut self.k;
        let dk_dls = &mut self.dk_dls;
        let diag_extra = sigma_sq + JITTER;

        let v_eta_sq = f64x4::splat(eta_sq);
        let v_half = f64x4::splat(0.5);
        let v_inv_ls_sq = f64x4::splat(inv_ls_sq);

        for i in 0..N {
            let v_xi = f64x4::splat(X_DATA[i]);

            // SIMD loop for j < i (in chunks of 4)
            let chunks = i / 4;
            let remainder_start = chunks * 4;

            for c in 0..chunks {
                let base = c * 4;
                let v_xj = f64x4::from_slice(&X_DATA[base..]);
                let v_d = v_xi - v_xj;
                let v_d_sq = v_d * v_d;
                let v_r_sq_scaled = v_d_sq * v_inv_ls_sq;

                // exp(-0.5 * r_sq_scaled) - need scalar fallback for exp
                let r_arr = (v_half * v_r_sq_scaled).to_array();
                let exp_arr = [
                    (-r_arr[0]).exp(),
                    (-r_arr[1]).exp(),
                    (-r_arr[2]).exp(),
                    (-r_arr[3]).exp(),
                ];

                for lane in 0..4 {
                    let j = base + lane;
                    let k_ij = eta_sq * exp_arr[lane];
                    let dk_ij = k_ij * v_r_sq_scaled.as_array()[lane];
                    k[i * N + j] = k_ij;
                    k[j * N + i] = k_ij;
                    dk_dls[i * N + j] = dk_ij;
                    dk_dls[j * N + i] = dk_ij;
                }
            }

            // Scalar remainder
            for j in remainder_start..i {
                let d = X_DATA[i] - X_DATA[j];
                let d_sq = d * d;
                let r_sq_scaled = d_sq * inv_ls_sq;
                let exp_term = (-0.5 * r_sq_scaled).exp();
                let k_ij = eta_sq * exp_term;
                let dk_ij = k_ij * r_sq_scaled;
                k[i * N + j] = k_ij;
                k[j * N + i] = k_ij;
                dk_dls[i * N + j] = dk_ij;
                dk_dls[j * N + i] = dk_ij;
            }

            // Diagonal
            k[i * N + i] = eta_sq + diag_extra;
            dk_dls[i * N + i] = 0.0;
        }

        // Cholesky (scalar - O(N³), hard to SIMD)
        for j in 0..N {
            let mut sum = k[j * N + j];
            for p in 0..j {
                sum -= k[j * N + p] * k[j * N + p];
            }
            if sum <= 0.0 {
                return Err(SampleError::Recoverable("Cholesky failed".to_string()));
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

        // Log determinant
        let mut log_det = 0.0;
        for i in 0..N {
            log_det += k[i * N + i].ln();
        }
        log_det *= 2.0;

        // Solve L z = y
        let alpha = &mut self.alpha;
        for i in 0..N {
            let mut sum = Y_DATA[i];
            for j in 0..i {
                sum -= k[i * N + j] * alpha[j];
            }
            alpha[i] = sum / k[i * N + i];
        }

        // Solve L^T alpha = z
        for i in (0..N).rev() {
            let mut sum = alpha[i];
            for j in (i + 1)..N {
                sum -= k[j * N + i] * alpha[j];
            }
            alpha[i] = sum / k[i * N + i];
        }

        // GP log-likelihood
        let mut yt_kinv_y = 0.0;
        for i in 0..N {
            yt_kinv_y += Y_DATA[i] * alpha[i];
        }
        logp += -0.5 * (N as f64 * LN_2PI + log_det + yt_kinv_y);

        // Compute K^{-1}
        let kinv = &mut self.kinv;
        let tmp = &mut self.tmp;
        for col in 0..N {
            for i in 0..N {
                let rhs = if i == col { 1.0 } else { 0.0 };
                let mut sum = rhs;
                for j in 0..i {
                    sum -= k[i * N + j] * tmp[j];
                }
                tmp[i] = sum / k[i * N + i];
            }
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

        // Gradients with SIMD for the O(N²) loops
        let mut grad_log_ls = 0.0;
        let mut grad_log_eta = 0.0;
        let mut grad_log_sigma = 0.0;

        let v_two_eta_sq = f64x4::splat(2.0 * eta_sq);

        for i in 0..N {
            let v_alpha_i = f64x4::splat(alpha[i]);
            let v_xi2 = f64x4::splat(X_DATA[i]);

            let mut v_grad_ls = f64x4::splat(0.0);
            let mut v_grad_eta = f64x4::splat(0.0);

            let chunks = N / 4;
            let rem_start = chunks * 4;

            for c in 0..chunks {
                let base = c * 4;
                // Load kinv row elements
                let v_kinv = f64x4::from_slice(&kinv[i * N + base..]);
                let v_alpha_j = f64x4::from_slice(&alpha[base..]);
                let v_w = v_kinv - v_alpha_i * v_alpha_j;

                // dk_dls
                let v_dk = f64x4::from_slice(&dk_dls[i * N + base..]);
                v_grad_ls += v_w * v_dk;

                // dk_d(log_eta) = 2*eta^2*exp(-0.5*d^2/ls^2)
                let v_xj2 = f64x4::from_slice(&X_DATA[base..]);
                let v_d2 = v_xi2 - v_xj2;
                let v_d_sq2 = v_d2 * v_d2;
                let v_r2 = v_d_sq2 * v_inv_ls_sq;
                let r_arr = (v_half * v_r2).to_array();
                let v_kernel = f64x4::from_array([
                    2.0 * eta_sq * (-r_arr[0]).exp(),
                    2.0 * eta_sq * (-r_arr[1]).exp(),
                    2.0 * eta_sq * (-r_arr[2]).exp(),
                    2.0 * eta_sq * (-r_arr[3]).exp(),
                ]);
                v_grad_eta += v_w * v_kernel;
            }

            grad_log_ls += v_grad_ls.reduce_sum();
            grad_log_eta += v_grad_eta.reduce_sum();

            // Scalar remainder + diagonal for sigma
            for j in rem_start..N {
                let w_ij = kinv[i * N + j] - alpha[i] * alpha[j];
                grad_log_ls += w_ij * dk_dls[i * N + j];
                let d = X_DATA[i] - X_DATA[j];
                let kernel_ij = eta_sq * (-0.5 * d * d * inv_ls_sq).exp();
                grad_log_eta += w_ij * 2.0 * kernel_ij;
            }

            // Sigma gradient (diagonal only)
            grad_log_sigma += (kinv[i * N + i] - alpha[i] * alpha[i]) * 2.0 * sigma_sq;
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
