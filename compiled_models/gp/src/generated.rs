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

const LN_2PI: f64 = 1.8378770664093453;
const JITTER: f64 = 1e-6;
const N: usize = 50;  // Y_N

pub const N_PARAMS: usize = 3;

#[derive(Storable, Clone)]
pub struct Draw {
    #[storable(dims("param"))]
    pub parameters: Vec<f64>,
}

pub struct GeneratedLogp {
    k_mat: Mat<f64>,      // N×N kernel matrix
    alpha: Mat<f64>,      // N×1 for K^{-1} y
    kinv: Mat<f64>,       // N×N for inverse
    dk_dls: Vec<f64>,     // N*N flat storage for dK/d(log_ls)
    dk_deta: Vec<f64>,    // N*N flat storage for dK/d(log_eta)  
    chol_buf: MemBuffer,  // scratch for cholesky_in_place
    inv_buf: MemBuffer,   // scratch for inverse
}

impl Default for GeneratedLogp {
    fn default() -> Self {
        let chol_scratch = factor::cholesky_in_place_scratch::<f64>(
            N, Par::Seq, Spec::<LltParams, f64>::default(),
        );
        let inv_scratch = inverse::inverse_scratch::<f64>(N, Par::Seq);
        Self {
            k_mat: Mat::zeros(N, N),
            alpha: Mat::zeros(N, 1),
            kinv: Mat::zeros(N, N),
            dk_dls: vec![0.0; N * N],
            dk_deta: vec![0.0; N * N],
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
        // Clear gradient
        for i in 0..N_PARAMS {
            gradient[i] = 0.0;
        }

        // Extract parameters (unconstrained space)
        let log_ls = position[0];
        let log_eta = position[1];
        let log_sigma = position[2];

        // Transform to constrained space
        let ls = log_ls.exp();
        let eta = log_eta.exp();
        let sigma = log_sigma.exp();

        // Precompute for efficiency
        let eta_sq = eta * eta;
        let inv_ls_sq = 1.0 / (ls * ls);
        let sigma_sq = sigma * sigma;

        // Build kernel matrix K and derivatives
        for i in 0..N {
            for j in 0..=i {
                let d = X_1_DATA[i] - X_1_DATA[j];
                let d_sq = d * d;
                let r_sq_scaled = d_sq * inv_ls_sq;
                let exp_term = (-0.5 * r_sq_scaled).exp();
                let k_ij = eta_sq * exp_term;

                // Store dK/d(log_ls) and dK/d(log_eta) for gradients
                self.dk_dls[i * N + j] = k_ij * r_sq_scaled;
                self.dk_dls[j * N + i] = self.dk_dls[i * N + j];
                
                self.dk_deta[i * N + j] = 2.0 * k_ij;
                self.dk_deta[j * N + i] = self.dk_deta[i * N + j];

                if i == j {
                    self.k_mat[(i, j)] = k_ij + sigma_sq + JITTER;
                } else {
                    self.k_mat[(i, j)] = k_ij;
                    self.k_mat[(j, i)] = k_ij;
                }
            }
        }

        // Cholesky decomposition
        factor::cholesky_in_place(
            self.k_mat.as_mut(),
            Default::default(),
            Par::Seq,
            MemStack::new(&mut self.chol_buf),
            Spec::<LltParams, f64>::default(),
        ).map_err(|_| {
            SampleError::Recoverable("Cholesky failed: not positive definite".to_string())
        })?;

        // Compute log determinant
        let mut log_det = 0.0;
        for i in 0..N {
            log_det += self.k_mat[(i, i)].ln();
        }
        log_det *= 2.0;

        // Solve K alpha = y
        for i in 0..N {
            self.alpha[(i, 0)] = Y_DATA[i];
        }
        solve::solve_in_place(
            self.k_mat.as_ref(),
            self.alpha.as_mut(),
            Par::Seq,
            MemStack::new(&mut []),
        );

        // Compute y^T K^{-1} y
        let mut quad_form = 0.0;
        for i in 0..N {
            quad_form += Y_DATA[i] * self.alpha[(i, 0)];
        }

        // GP log-likelihood (y ~ MvNormal(0, K))
        let gp_logp = -0.5 * (N as f64 * LN_2PI + log_det + quad_form);

        // Prior log-probabilities (HalfNormal with scale=5, log-transformed)
        // p(ls) ~ HalfNormal(0, 5) with LogTransform
        // logp = log(2) - 0.5*log(2*π) - log(5) - 0.5*(ls/5)^2 + log(ls)
        let scale = 5.0_f64;
        let inv_scale_sq = 1.0 / (scale * scale);
        let ls_logp = (2.0_f64).ln() - 0.5 * LN_2PI - scale.ln() - 0.5 * ls * ls * inv_scale_sq + log_ls;
        let eta_logp = (2.0_f64).ln() - 0.5 * LN_2PI - scale.ln() - 0.5 * eta * eta * inv_scale_sq + log_eta;
        let sigma_logp = (2.0_f64).ln() - 0.5 * LN_2PI - scale.ln() - 0.5 * sigma * sigma * inv_scale_sq + log_sigma;

        let total_logp = gp_logp + ls_logp + eta_logp + sigma_logp;

        // Compute K^{-1} for gradients
        inverse::inverse(
            self.kinv.as_mut(),
            self.k_mat.as_ref(),
            Par::Seq,
            MemStack::new(&mut self.inv_buf),
        );

        // GP gradients w.r.t. hyperparameters
        let mut grad_log_ls = 0.0;
        let mut grad_log_eta = 0.0;
        let mut grad_log_sigma = 0.0;

        for i in 0..N {
            let alpha_i = self.alpha[(i, 0)];
            for j in 0..N {
                let alpha_j = self.alpha[(j, 0)];
                let kinv_ij = if i >= j { self.kinv[(i, j)] } else { self.kinv[(j, i)] };
                let w_ij = kinv_ij - alpha_i * alpha_j;
                
                grad_log_ls += w_ij * self.dk_dls[i * N + j];
                grad_log_eta += w_ij * self.dk_deta[i * N + j];
                
                // dK/d(log_sigma) = 2 * sigma^2 * delta_ij
                if i == j {
                    grad_log_sigma += w_ij * 2.0 * sigma_sq;
                }
            }
        }
        grad_log_ls *= -0.5;
        grad_log_eta *= -0.5;
        grad_log_sigma *= -0.5;

        // Prior gradients w.r.t. log-parameters
        // d(logp)/d(log_x) = -x^2/scale^2 + 1 for HalfNormal with LogTransform
        let ls_grad_prior = -ls * ls * inv_scale_sq + 1.0;
        let eta_grad_prior = -eta * eta * inv_scale_sq + 1.0;
        let sigma_grad_prior = -sigma * sigma * inv_scale_sq + 1.0;

        // Total gradients
        gradient[0] += grad_log_ls + ls_grad_prior;
        gradient[1] += grad_log_eta + eta_grad_prior;
        gradient[2] += grad_log_sigma + sigma_grad_prior;

        Ok(total_logp)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self, _rng: &mut R, array: &[f64],
    ) -> Result<Draw, CpuMathError> {
        Ok(Draw { parameters: array.to_vec() })
    }
}