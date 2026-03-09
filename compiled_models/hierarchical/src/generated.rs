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

pub const N_PARAMS: usize = 12;

#[derive(Storable, Clone)]
pub struct Draw {
    #[storable(dims("param"))]
    pub parameters: Vec<f64>,
}

#[derive(Clone)]
pub struct GeneratedLogp;

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
        // Parameter extraction (unconstrained space)
        let mu_a = position[0];
        let log_sigma_a = position[1];
        let a_offset = &position[2..10]; // shape [8]
        let b = position[10];
        let log_sigma_y = position[11];
        
        // Transform to constrained space
        let sigma_a = log_sigma_a.exp();
        let sigma_y = log_sigma_y.exp();
        
        // Precompute constants and invariants
        const LN_2PI: f64 = 1.8378770664093453;
        
        let mut logp = 0.0;
        
        // Initialize gradient
        gradient.fill(0.0);
        
        // mu_a ~ Normal(0, 10)
        // logp = -0.5*log(2*pi) - log(10) - 0.5*((mu_a - 0)/10)^2
        let mu_a_normalized = mu_a * 0.1;
        logp += -0.5 * LN_2PI - 10.0_f64.ln() - 0.5 * mu_a_normalized * mu_a_normalized;
        gradient[0] += -mu_a * 0.01; // d/d(mu_a) = -(mu_a - 0) / 10^2
        
        // sigma_a ~ HalfNormal(5) with LogTransform
        // x = exp(log_sigma_a), logp = log(2) - 0.5*log(2*pi) - log(5) - 0.5*(x/5)^2 + log_sigma_a
        let sigma_a_normalized = sigma_a * 0.2;
        logp += 2.0_f64.ln() - 0.5 * LN_2PI - 5.0_f64.ln() - 0.5 * sigma_a_normalized * sigma_a_normalized + log_sigma_a;
        gradient[1] += -sigma_a * sigma_a * 0.04 + 1.0; // d/d(log_sigma_a)
        
        // a_offset ~ Normal(0, 1, shape=8)
        // logp = sum over i of [-0.5*log(2*pi) - 0.5*a_offset[i]^2]
        for i in 0..8 {
            logp += -0.5 * LN_2PI - 0.5 * a_offset[i] * a_offset[i];
            gradient[2 + i] += -a_offset[i]; // d/d(a_offset[i])
        }
        
        // b ~ Normal(0, 10)
        let b_normalized = b * 0.1;
        logp += -0.5 * LN_2PI - 10.0_f64.ln() - 0.5 * b_normalized * b_normalized;
        gradient[10] += -b * 0.01; // d/d(b)
        
        // sigma_y ~ HalfNormal(5) with LogTransform
        let sigma_y_normalized = sigma_y * 0.2;
        logp += 2.0_f64.ln() - 0.5 * LN_2PI - 5.0_f64.ln() - 0.5 * sigma_y_normalized * sigma_y_normalized + log_sigma_y;
        gradient[11] += -sigma_y * sigma_y * 0.04 + 1.0; // d/d(log_sigma_y)
        
        // Precompute for observations: a = mu_a + sigma_a * a_offset
        let mut a = [0.0; 8];
        for i in 0..8 {
            a[i] = mu_a + sigma_a * a_offset[i];
        }
        
        // y ~ Normal(a[group_idx] + b * x, sigma_y)
        // Precompute invariants outside the loop
        let inv_sigma_y = 1.0 / sigma_y;
        let inv_sigma_y_sq = inv_sigma_y * inv_sigma_y;
        let log_norm = -0.5 * LN_2PI - log_sigma_y;
        
        // Initialize accumulators for efficient gradient computation
        let mut grad_mu_a = 0.0;
        let mut grad_b = 0.0;
        let mut grad_log_sigma_y = 0.0;
        let mut grad_a_offset = [0.0; 8];
        
        for i in 0..Y_N {
            let group_idx = X_1_DATA[i] as usize;
            let x_val = X_0_DATA[i];
            let y_val = Y_DATA[i];
            
            let mu_i = a[group_idx] + b * x_val;
            let residual = y_val - mu_i;
            let residual_scaled = residual * inv_sigma_y_sq;
            
            // Add to logp
            logp += log_norm - 0.5 * residual * residual * inv_sigma_y_sq;
            
            // Accumulate gradients
            grad_mu_a += residual_scaled;
            grad_b += residual_scaled * x_val;
            grad_log_sigma_y += residual * residual * inv_sigma_y_sq - 1.0;
            grad_a_offset[group_idx] += residual_scaled * sigma_a;
        }
        
        // Add accumulated gradients
        gradient[0] += grad_mu_a;
        gradient[10] += grad_b;
        gradient[11] += grad_log_sigma_y;
        
        // For sigma_a gradient: d/d(log_sigma_a) includes chain rule
        let mut grad_log_sigma_a = 0.0;
        for i in 0..8 {
            grad_log_sigma_a += grad_a_offset[i] * a_offset[i];
            gradient[2 + i] += grad_a_offset[i]; // d/d(a_offset[i])
        }
        gradient[1] += grad_log_sigma_a;
        
        Ok(logp)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self, _rng: &mut R, array: &[f64],
    ) -> Result<Draw, CpuMathError> {
        Ok(Draw { parameters: array.to_vec() })
    }
}