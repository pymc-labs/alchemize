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
const LN_2PI: f64 = 1.8378770664093453;

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
        let mu = position[0];
        let log_sigma = position[1];
        let sigma = log_sigma.exp();
        
        // Check sigma > 0 (should always be true for exp, but keeping for safety)
        if sigma <= 0.0 {
            return Err(SampleError::Recoverable("sigma must be positive".to_string()));
        }
        
        let mut logp = 0.0;
        gradient[0] = 0.0;
        gradient[1] = 0.0;
        
        // Prior: mu ~ Normal(0, 10)
        // logp = -0.5*log(2*pi) - log(10) - 0.5*((mu-0)/10)^2
        let mu_prior_logp = -0.5 * LN_2PI - 10.0f64.ln() - 0.5 * (mu / 10.0).powi(2);
        logp += mu_prior_logp;
        
        // Gradient for mu prior
        let grad_mu_prior = -mu / 100.0;  // d/dmu of -0.5*(mu/10)^2 = -mu/100
        gradient[0] += grad_mu_prior;
        
        // Prior: sigma ~ HalfNormal(5) with LogTransform
        // sigma = exp(log_sigma)
        // HalfNormal(sigma | 5): logp = log(2) - 0.5*log(2*pi) - log(5) - 0.5*(sigma/5)^2
        // With LogTransform Jacobian: +log_sigma
        let sigma_prior_logp = 2.0f64.ln() - 0.5 * LN_2PI - 5.0f64.ln() - 0.5 * (sigma / 5.0).powi(2) + log_sigma;
        logp += sigma_prior_logp;
        
        // Gradient for sigma prior (w.r.t. log_sigma)
        // d/d(log_sigma) = -sigma^2/25 + 1
        let grad_sigma_prior = -sigma * sigma / 25.0 + 1.0;
        gradient[1] += grad_sigma_prior;
        
        // Likelihood: y ~ Normal(mu, sigma)
        // For each observation: logp = -0.5*log(2*pi) - log(sigma) - 0.5*((y[i] - mu)/sigma)^2
        
        // Precompute constants for efficiency
        let inv_sigma = 1.0 / sigma;
        let inv_sigma_sq = inv_sigma * inv_sigma;
        let log_norm_term = -0.5 * LN_2PI - log_sigma;  // -0.5*log(2*pi) - log(sigma)
        
        let mut grad_mu_obs = 0.0;
        let mut grad_log_sigma_obs = 0.0;
        
        for i in 0..Y_N {
            let y_i = Y_DATA[i];
            let residual = y_i - mu;
            
            // Logp contribution from this observation
            let obs_logp = log_norm_term - 0.5 * residual * residual * inv_sigma_sq;
            logp += obs_logp;
            
            // Gradient contributions
            // d/dmu = (y_i - mu) / sigma^2
            grad_mu_obs += residual * inv_sigma_sq;
            
            // d/d(log_sigma) = -1 + (y_i - mu)^2 / sigma^2
            grad_log_sigma_obs += -1.0 + residual * residual * inv_sigma_sq;
        }
        
        gradient[0] += grad_mu_obs;
        gradient[1] += grad_log_sigma_obs;
        
        Ok(logp)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self, _rng: &mut R, array: &[f64],
    ) -> Result<Draw, CpuMathError> {
        Ok(Draw { parameters: array.to_vec() })
    }
}