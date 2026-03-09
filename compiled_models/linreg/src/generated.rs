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

pub const N_PARAMS: usize = 3;

const LN_2PI: f64 = 1.8378770664093453;  // ln(2*pi)

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
        // Extract parameters
        let alpha = position[0];
        let beta = position[1];
        let log_sigma = position[2];
        
        // Transform sigma from log space
        let sigma = log_sigma.exp();
        
        // Initialize gradient
        gradient[0] = 0.0;
        gradient[1] = 0.0;
        gradient[2] = 0.0;
        
        let mut logp = 0.0;
        
        // Prior for alpha ~ Normal(0, 10)
        // logp = -0.5*ln(2*pi) - ln(10) - 0.5*(alpha/10)^2
        let alpha_scaled = alpha * 0.1; // alpha / 10
        logp += -0.5 * LN_2PI - 10.0f64.ln() - 0.5 * alpha_scaled * alpha_scaled;
        gradient[0] += -alpha * 0.01; // d/d_alpha = -alpha/100
        
        // Prior for beta ~ Normal(0, 10)  
        // logp = -0.5*ln(2*pi) - ln(10) - 0.5*(beta/10)^2
        let beta_scaled = beta * 0.1; // beta / 10
        logp += -0.5 * LN_2PI - 10.0f64.ln() - 0.5 * beta_scaled * beta_scaled;
        gradient[1] += -beta * 0.01; // d/d_beta = -beta/100
        
        // Prior for sigma ~ HalfNormal(5) with LogTransform
        // HalfNormal(sigma | 5) = 2 * Normal(sigma | 0, 5) for sigma > 0
        // logp = ln(2) - 0.5*ln(2*pi) - ln(5) - 0.5*(sigma/5)^2 + log_sigma (Jacobian)
        if sigma <= 0.0 {
            return Err(SampleError::Recoverable("sigma must be positive".to_string()));
        }
        let sigma_scaled = sigma * 0.2; // sigma / 5
        logp += 2.0f64.ln() - 0.5 * LN_2PI - 5.0f64.ln() - 0.5 * sigma_scaled * sigma_scaled + log_sigma;
        // d/d_log_sigma = -sigma^2/25 + 1 = -0.04*sigma^2 + 1
        gradient[2] += -0.04 * sigma * sigma + 1.0;
        
        // Likelihood for y ~ Normal(alpha + beta * x, sigma)
        // Precompute constants for efficiency
        let inv_sigma = 1.0 / sigma;
        let inv_sigma_sq = inv_sigma * inv_sigma;
        let neg_log_sigma = -log_sigma;  // -ln(sigma) = -log_sigma
        let log_norm_const = -0.5 * LN_2PI + neg_log_sigma;  // constant per observation
        
        let mut grad_alpha_acc = 0.0;
        let mut grad_beta_acc = 0.0;
        let mut grad_log_sigma_acc = 0.0;
        
        for i in 0..Y_N {
            let y_i = Y_DATA[i];
            let x_i = X_0_DATA[i];
            let mu_i = alpha + beta * x_i;
            let residual = y_i - mu_i;
            let residual_scaled = residual * inv_sigma;
            
            // Add log probability for this observation
            logp += log_norm_const - 0.5 * residual_scaled * residual_scaled;
            
            // Gradients
            let scaled_residual = residual * inv_sigma_sq;
            grad_alpha_acc += scaled_residual;
            grad_beta_acc += scaled_residual * x_i;
            // d/d_log_sigma = residual^2/sigma^2 - 1
            grad_log_sigma_acc += residual_scaled * residual_scaled - 1.0;
        }
        
        gradient[0] += grad_alpha_acc;
        gradient[1] += grad_beta_acc;
        gradient[2] += grad_log_sigma_acc;
        
        Ok(logp)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self, _rng: &mut R, array: &[f64],
    ) -> Result<Draw, CpuMathError> {
        Ok(Draw { parameters: array.to_vec() })
    }
}