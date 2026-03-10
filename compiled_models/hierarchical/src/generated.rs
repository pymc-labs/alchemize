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

// Mathematical constants
const LN_2PI: f64 = 1.8378770664093453;

#[derive(Storable, Clone)]
pub struct Draw {
    #[storable(dims("param"))]
    pub parameters: Vec<f64>,
}

#[derive(Clone, Default)]
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
        // Zero out gradient
        gradient.fill(0.0);
        
        // Extract parameters (in unconstrained space)
        let mu_a = position[0];
        let log_sigma_a = position[1];
        let a_offset = &position[2..10]; // length 8
        let b = position[10];
        let log_sigma_y = position[11];
        
        // Transform log parameters
        let sigma_a = log_sigma_a.exp();
        let sigma_y = log_sigma_y.exp();
        
        let mut logp = 0.0;
        
        // Prior: mu_a ~ Normal(0, 10)
        // logp = -0.5*log(2*pi) - log(10) - 0.5*(mu_a/10)^2
        logp += -0.5 * LN_2PI - (10.0_f64).ln() - 0.5 * (mu_a * 0.1) * (mu_a * 0.1);
        gradient[0] += -(mu_a * 0.01); // d/d(mu_a) = -mu_a/100
        
        // Prior: sigma_a ~ HalfNormal(5) with LogTransform
        // x = exp(log_x), logp = log(2) - 0.5*log(2*pi) - log(5) - 0.5*(x/5)^2 + log_x
        if sigma_a > 0.0 {
            logp += (2.0_f64).ln() - 0.5 * LN_2PI - (5.0_f64).ln() - 0.5 * (sigma_a * 0.2) * (sigma_a * 0.2) + log_sigma_a;
            gradient[1] += -sigma_a * sigma_a * 0.04 + 1.0; // d/d(log_sigma_a) = -sigma_a^2/25 + 1
        } else {
            return Ok(f64::NEG_INFINITY);
        }
        
        // Prior: a_offset ~ Normal(0, 1) for each component
        for i in 0..8 {
            logp += -0.5 * LN_2PI - 0.5 * a_offset[i] * a_offset[i];
            gradient[2 + i] += -a_offset[i]; // d/d(a_offset[i]) = -a_offset[i]
        }
        
        // Prior: b ~ Normal(0, 10)
        logp += -0.5 * LN_2PI - (10.0_f64).ln() - 0.5 * (b * 0.1) * (b * 0.1);
        gradient[10] += -(b * 0.01); // d/d(b) = -b/100
        
        // Prior: sigma_y ~ HalfNormal(5) with LogTransform
        if sigma_y > 0.0 {
            logp += (2.0_f64).ln() - 0.5 * LN_2PI - (5.0_f64).ln() - 0.5 * (sigma_y * 0.2) * (sigma_y * 0.2) + log_sigma_y;
            gradient[11] += -sigma_y * sigma_y * 0.04 + 1.0; // d/d(log_sigma_y) = -sigma_y^2/25 + 1
        } else {
            return Ok(f64::NEG_INFINITY);
        }
        
        // Likelihood: y ~ Normal(mu, sigma_y)
        // mu[i] = (mu_a + sigma_a * a_offset[group[i]]) + b * X_0[i]
        
        // Precompute terms for efficiency
        let inv_sigma_y_sq = 1.0 / (sigma_y * sigma_y);
        let log_norm_y = -0.5 * LN_2PI - log_sigma_y;
        
        // Group gradient accumulators
        let mut grad_a_offset = [0.0f64; 8];
        let mut grad_mu_a = 0.0f64;
        let mut grad_b = 0.0f64;
        let mut grad_log_sigma_y = 0.0f64;
        let mut grad_log_sigma_a = 0.0f64;
        
        for i in 0..Y_N {
            let group = X_1_DATA[i] as usize;
            let x = X_0_DATA[i];
            let y = Y_DATA[i];
            
            // Compute mean: mu = mu_a + sigma_a * a_offset[group] + b * x
            let mu = mu_a + sigma_a * a_offset[group] + b * x;
            let residual = y - mu;
            
            // Add to logp
            logp += log_norm_y - 0.5 * residual * residual * inv_sigma_y_sq;
            
            // Gradient accumulation
            let scaled_residual = residual * inv_sigma_y_sq;
            grad_mu_a += scaled_residual;
            grad_a_offset[group] += scaled_residual * sigma_a;
            grad_b += scaled_residual * x;
            grad_log_sigma_y += residual * residual * inv_sigma_y_sq - 1.0;
            
            // Gradient w.r.t. log_sigma_a: d/d(log_sigma_a) = d/d(sigma_a) * sigma_a
            // d/d(sigma_a) of likelihood = scaled_residual * a_offset[group]
            grad_log_sigma_a += scaled_residual * a_offset[group] * sigma_a;
        }
        
        // Add to gradients
        gradient[0] += grad_mu_a;
        gradient[1] += grad_log_sigma_a;
        
        for g in 0..8 {
            gradient[2 + g] += grad_a_offset[g];
        }
        gradient[10] += grad_b;
        gradient[11] += grad_log_sigma_y;
        
        Ok(logp)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self, _rng: &mut R, array: &[f64],
    ) -> Result<Draw, CpuMathError> {
        Ok(Draw { parameters: array.to_vec() })
    }
}