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

pub const N_PARAMS: usize = 2;

// Constants
const LN_2PI: f64 = 1.8378770664093453; // ln(2*pi)

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
        let mu = position[0];
        let log_sigma = position[1];
        let sigma = log_sigma.exp();
        
        // Initialize gradient
        gradient[0] = 0.0;
        gradient[1] = 0.0;
        
        let mut logp_total = 0.0;
        
        // Prior for mu ~ Normal(0, 10)
        // logp = -0.5*log(2*pi) - log(10) - 0.5*((mu - 0)/10)^2
        let mu_prior_logp = -0.5 * LN_2PI - (10.0_f64).ln() - 0.5 * (mu / 10.0).powi(2);
        logp_total += mu_prior_logp;
        
        // Gradient for mu prior: d/d_mu = -(mu - 0)/(10^2) = -mu/100
        gradient[0] += -mu / 100.0;
        
        // Prior for sigma ~ HalfNormal(0, 5) with LogTransform
        // The prior is on sigma = exp(log_sigma)
        // HalfNormal(sigma | 5): logp = log(2) - 0.5*log(2*pi) - log(5) - 0.5*(sigma/5)^2
        // Plus Jacobian: + log_sigma
        let sigma_scaled = sigma / 5.0;
        let half_normal_logp = 2.0_f64.ln() - 0.5 * LN_2PI - 5.0_f64.ln() - 0.5 * sigma_scaled * sigma_scaled;
        let sigma_prior_logp = half_normal_logp + log_sigma; // Jacobian adjustment
        logp_total += sigma_prior_logp;
        
        // Gradient for sigma prior w.r.t. log_sigma
        // d/d_log_sigma = -sigma^2/25 + 1
        gradient[1] += -sigma * sigma / 25.0 + 1.0;
        
        // Likelihood for y ~ Normal(mu, sigma)
        // For each observation: logp = -0.5*log(2*pi) - log(sigma) - 0.5*((y[i] - mu)/sigma)^2
        
        let inv_sigma = 1.0 / sigma;
        let inv_sigma_sq = inv_sigma * inv_sigma;
        let neg_log_sigma = -log_sigma; // Since sigma = exp(log_sigma), log(sigma) = log_sigma
        let log_norm_constant = -0.5 * LN_2PI + neg_log_sigma;
        
        let mut grad_mu_accum = 0.0;
        let mut grad_log_sigma_accum = 0.0;
        
        for i in 0..Y_N {
            let residual = Y_DATA[i] - mu;
            let residual_scaled = residual * inv_sigma;
            
            // Likelihood contribution
            logp_total += log_norm_constant - 0.5 * residual_scaled * residual_scaled;
            
            // Gradients
            let residual_scaled_sq = residual_scaled * residual_scaled;
            
            // d/d_mu = (y[i] - mu)/sigma^2 = residual * inv_sigma_sq
            grad_mu_accum += residual * inv_sigma_sq;
            
            // d/d_log_sigma = -1 + (y[i] - mu)^2/sigma^2 = -1 + residual_scaled^2
            grad_log_sigma_accum += -1.0 + residual_scaled_sq;
        }
        
        gradient[0] += grad_mu_accum;
        gradient[1] += grad_log_sigma_accum;
        
        Ok(logp_total)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self, _rng: &mut R, array: &[f64],
    ) -> Result<Draw, CpuMathError> {
        Ok(Draw { parameters: array.to_vec() })
    }
}