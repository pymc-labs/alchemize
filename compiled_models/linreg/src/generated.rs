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
        // Extract parameters
        let alpha = position[0];
        let beta = position[1]; 
        let log_sigma = position[2];  // sigma_log__ in unconstrained space
        
        // Transform log_sigma to sigma
        let sigma = log_sigma.exp();
        
        // Initialize logp and gradients
        let mut logp = 0.0;
        gradient[0] = 0.0; // grad_alpha
        gradient[1] = 0.0; // grad_beta  
        gradient[2] = 0.0; // grad_log_sigma
        
        // Prior for alpha: Normal(0, 10)
        // logp = -0.5*ln(2π) - ln(10) - 0.5*(alpha/10)^2
        // The constant -3.221523658174155 = -0.5*ln(2π) - ln(10)
        let alpha_scaled = alpha * 0.1; // alpha / 10
        logp += -3.221523658174155 - 0.5 * alpha_scaled * alpha_scaled;
        gradient[0] += -alpha * 0.01; // -alpha / 100 = derivative
        
        // Prior for beta: Normal(0, 10) - identical to alpha
        let beta_scaled = beta * 0.1;
        logp += -3.221523658174155 - 0.5 * beta_scaled * beta_scaled;  
        gradient[1] += -beta * 0.01;
        
        // Prior for sigma: HalfNormal(5) with LogTransform
        // The prior on sigma is HalfNormal(5), but we need Jacobian adjustment
        // HalfNormal(5) logp = ln(2) - 0.5*ln(2π) - ln(5) - 0.5*(sigma/5)^2
        // Plus Jacobian +ln(sigma) = +log_sigma for the LogTransform
        // The constant -1.83522929514961 comes from ln(2) - 0.5*ln(2π) - ln(5)
        if sigma > 0.0 {
            let sigma_scaled = sigma * 0.2; // sigma / 5
            logp += -1.83522929514961 - 0.5 * sigma_scaled * sigma_scaled + log_sigma;
            // Gradient w.r.t. log_sigma: d/d(log_sigma) = -sigma^2/25 + 1
            gradient[2] += -sigma * sigma * 0.04 + 1.0; // -sigma^2 / 25 + 1
        } else {
            return Err(SampleError::Recoverable("sigma <= 0".to_string()));
        }
        
        // Likelihood: y ~ Normal(mu, sigma) where mu = alpha + beta * x
        // For each observation: logp += -0.5*ln(2π) - ln(sigma) - 0.5*((y - mu)/sigma)^2
        
        // Precompute constants for efficiency  
        let inv_sigma = 1.0 / sigma;
        let inv_sigma_sq = inv_sigma * inv_sigma;
        let neg_half_ln_2pi = -0.5 * LN_2PI; // -0.9189385332046727
        let neg_log_sigma = -log_sigma; // -ln(sigma)
        
        // Accumulate gradients efficiently
        let mut grad_alpha_acc = 0.0;
        let mut grad_beta_acc = 0.0; 
        let mut grad_log_sigma_acc = 0.0;
        
        for i in 0..Y_N {
            let y_i = Y_DATA[i];
            let x_i = X_0_DATA[i];
            let mu_i = alpha + beta * x_i;
            let residual = y_i - mu_i;
            let residual_scaled = residual * inv_sigma;
            
            // Logp contribution from this observation
            logp += neg_half_ln_2pi + neg_log_sigma - 0.5 * residual_scaled * residual_scaled;
            
            // Gradient contributions
            let grad_factor = residual * inv_sigma_sq;
            grad_alpha_acc += grad_factor;
            grad_beta_acc += grad_factor * x_i;
            
            // For log_sigma: d/d(log_sigma) = -1 + residual^2/sigma^2 
            grad_log_sigma_acc += -1.0 + residual_scaled * residual_scaled;
        }
        
        // Add accumulated gradients
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