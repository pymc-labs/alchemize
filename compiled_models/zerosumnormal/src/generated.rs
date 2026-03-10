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

pub const N_PARAMS: usize = 124;
const LN_2PI: f64 = 1.8378770664093453;

#[derive(Storable, Clone)]
pub struct Draw {
    #[storable(dims("param"))]
    pub parameters: Vec<f64>,
}

#[derive(Clone)]
pub struct GeneratedLogp {
    // Pre-allocate arrays for ZeroSumTransform operations
    store_effect_full: [f64; 6],
    day_effect_full: [f64; 7], 
    interaction_temp: Vec<f64>,  // 6*7*3 = 126 elements
    interaction_full: Vec<f64>,  // 6*7*4 = 168 elements
}

impl Default for GeneratedLogp {
    fn default() -> Self {
        Self {
            store_effect_full: [0.0; 6],
            day_effect_full: [0.0; 7],
            interaction_temp: vec![0.0; 6 * 7 * 3], // 126
            interaction_full: vec![0.0; 6 * 7 * 4], // 168
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
        // Zero gradient
        gradient.fill(0.0);
        
        let mut logp = 0.0f64;
        
        // Extract parameters from position
        let grand_mean = position[0];
        let log_sigma_store = position[1];
        let log_sigma_day = position[2]; 
        let log_sigma_cat = position[3];
        let store_effect_unc = &position[4..9];   // 5 elements
        let day_effect_unc = &position[9..15];   // 6 elements
        let interaction_unc = &position[15..123]; // 108 elements (6*6*3)
        let log_sigma_y = position[123];

        // Transform log parameters to constrained space
        let sigma_store = log_sigma_store.exp();
        let sigma_day = log_sigma_day.exp(); 
        let sigma_cat = log_sigma_cat.exp();
        let sigma_y = log_sigma_y.exp();

        // === PRIOR TERMS ===
        
        // grand_mean ~ Normal(0, 10)
        let grand_mean_term = -0.5 * LN_2PI - (10.0f64).ln() - 0.5 * (grand_mean / 10.0).powi(2);
        logp += grand_mean_term;
        gradient[0] += -grand_mean / 100.0; // d/d(grand_mean)

        // sigma_store ~ HalfNormal(2) with LogTransform
        if sigma_store > 0.0 {
            let hn_term = (2.0f64).ln() - 0.5 * LN_2PI - (2.0f64).ln() - 0.5 * (sigma_store / 2.0).powi(2) + log_sigma_store;
            logp += hn_term;
            gradient[1] += -sigma_store * sigma_store / 4.0 + 1.0; // d/d(log_sigma_store)
        } else {
            return Ok(f64::NEG_INFINITY);
        }

        // sigma_day ~ HalfNormal(2) with LogTransform  
        if sigma_day > 0.0 {
            let hn_term = (2.0f64).ln() - 0.5 * LN_2PI - (2.0f64).ln() - 0.5 * (sigma_day / 2.0).powi(2) + log_sigma_day;
            logp += hn_term;
            gradient[2] += -sigma_day * sigma_day / 4.0 + 1.0; // d/d(log_sigma_day)
        } else {
            return Ok(f64::NEG_INFINITY);
        }

        // sigma_cat ~ HalfNormal(2) with LogTransform
        if sigma_cat > 0.0 {
            let hn_term = (2.0f64).ln() - 0.5 * LN_2PI - (2.0f64).ln() - 0.5 * (sigma_cat / 2.0).powi(2) + log_sigma_cat;
            logp += hn_term;
            gradient[3] += -sigma_cat * sigma_cat / 4.0 + 1.0; // d/d(log_sigma_cat)
        } else {
            return Ok(f64::NEG_INFINITY);
        }

        // sigma_y ~ HalfNormal(5) with LogTransform
        if sigma_y > 0.0 {
            let hn_term = (2.0f64).ln() - 0.5 * LN_2PI - (5.0f64).ln() - 0.5 * (sigma_y / 5.0).powi(2) + log_sigma_y;
            logp += hn_term;
            gradient[123] += -sigma_y * sigma_y / 25.0 + 1.0; // d/d(log_sigma_y)
        } else {
            return Ok(f64::NEG_INFINITY);
        }

        // store_effect ~ ZeroSumNormal(sigma_store, shape=[6])
        // Uses 5 unconstrained parameters, logp evaluated on unconstrained values only
        let n_store_unc = 5.0_f64;
        let inv_sigma_store_sq = 1.0 / (sigma_store * sigma_store);
        let log_term_store = n_store_unc * (-0.5 * LN_2PI - log_sigma_store);
        logp += log_term_store;
        
        let mut sum_sq_store = 0.0;
        for i in 0..5 {
            sum_sq_store += store_effect_unc[i] * store_effect_unc[i];
            gradient[4 + i] += -store_effect_unc[i] * inv_sigma_store_sq;
        }
        logp += -0.5 * sum_sq_store * inv_sigma_store_sq;
        gradient[1] += (-n_store_unc + sum_sq_store / (sigma_store * sigma_store));

        // day_effect ~ ZeroSumNormal(sigma_day, shape=[7])
        // Uses 6 unconstrained parameters
        let n_day_unc = 6.0_f64;
        let inv_sigma_day_sq = 1.0 / (sigma_day * sigma_day);
        let log_term_day = n_day_unc * (-0.5 * LN_2PI - log_sigma_day);
        logp += log_term_day;
        
        let mut sum_sq_day = 0.0;
        for i in 0..6 {
            sum_sq_day += day_effect_unc[i] * day_effect_unc[i];
            gradient[9 + i] += -day_effect_unc[i] * inv_sigma_day_sq;
        }
        logp += -0.5 * sum_sq_day * inv_sigma_day_sq;
        gradient[2] += (-n_day_unc + sum_sq_day / (sigma_day * sigma_day));

        // interaction ~ ZeroSumNormal(sigma_cat, shape=[6,7,4]) with zerosum_axes=[-2,-1]
        // Unconstrained shape: [6,6,3] = 108 elements
        let n_interaction_unc = 108.0_f64;
        let inv_sigma_cat_sq = 1.0 / (sigma_cat * sigma_cat);
        let log_term_interaction = n_interaction_unc * (-0.5 * LN_2PI - log_sigma_cat);
        logp += log_term_interaction;
        
        let mut sum_sq_interaction = 0.0;
        for i in 0..108 {
            sum_sq_interaction += interaction_unc[i] * interaction_unc[i];
            gradient[15 + i] += -interaction_unc[i] * inv_sigma_cat_sq;
        }
        logp += -0.5 * sum_sq_interaction * inv_sigma_cat_sq;
        gradient[3] += (-n_interaction_unc + sum_sq_interaction / (sigma_cat * sigma_cat));

        // === FORWARD TRANSFORMS FOR LIKELIHOOD ===
        
        // Transform store_effect: 5 -> 6 elements
        let sum_x_store: f64 = store_effect_unc.iter().sum();
        let n_store = 6.0_f64;
        let norm_store = sum_x_store / (n_store.sqrt() + n_store);
        let fill_store = norm_store - sum_x_store / n_store.sqrt();
        
        for i in 0..5 {
            self.store_effect_full[i] = store_effect_unc[i] - norm_store;
        }
        self.store_effect_full[5] = fill_store - norm_store;

        // Transform day_effect: 6 -> 7 elements
        let sum_x_day: f64 = day_effect_unc.iter().sum();
        let n_day = 7.0_f64;
        let norm_day = sum_x_day / (n_day.sqrt() + n_day);
        let fill_day = norm_day - sum_x_day / n_day.sqrt();
        
        for i in 0..6 {
            self.day_effect_full[i] = day_effect_unc[i] - norm_day;
        }
        self.day_effect_full[6] = fill_day - norm_day;

        // Transform interaction: (6,6,3) -> (6,7,3) -> (6,7,4)
        // First extend axis -2: 6->7 along dim 1
        for i in 0..6 {
            for k in 0..3 {
                let mut sum_j = 0.0;
                for j in 0..6 {
                    sum_j += interaction_unc[i * 6 * 3 + j * 3 + k];
                }
                let n_j = 7.0_f64;
                let norm_j = sum_j / (n_j.sqrt() + n_j);
                let fill_j = norm_j - sum_j / n_j.sqrt();
                
                for j in 0..6 {
                    self.interaction_temp[i * 7 * 3 + j * 3 + k] = 
                        interaction_unc[i * 6 * 3 + j * 3 + k] - norm_j;
                }
                self.interaction_temp[i * 7 * 3 + 6 * 3 + k] = fill_j - norm_j;
            }
        }
        
        // Then extend axis -1: 3->4 along dim 2
        for i in 0..6 {
            for j in 0..7 {
                let mut sum_k = 0.0;
                for k in 0..3 {
                    sum_k += self.interaction_temp[i * 7 * 3 + j * 3 + k];
                }
                let n_k = 4.0_f64;
                let norm_k = sum_k / (n_k.sqrt() + n_k);
                let fill_k = norm_k - sum_k / n_k.sqrt();
                
                for k in 0..3 {
                    self.interaction_full[i * 7 * 4 + j * 4 + k] = 
                        self.interaction_temp[i * 7 * 3 + j * 3 + k] - norm_k;
                }
                self.interaction_full[i * 7 * 4 + j * 4 + 3] = fill_k - norm_k;
            }
        }

        // === OBSERVED LIKELIHOOD ===
        // y ~ Normal(mu, sigma_y) where mu = grand_mean + store_effect + day_effect + interaction
        
        let inv_sigma_y_sq = 1.0 / (sigma_y * sigma_y);
        let neg_log_sigma_y = -log_sigma_y;
        let log_norm_y = -0.5 * LN_2PI + neg_log_sigma_y;
        
        let mut grad_grand_mean = 0.0;
        let mut grad_log_sigma_y = 0.0;
        let mut grad_store_effect = [0.0; 6];
        let mut grad_day_effect = [0.0; 7];
        let mut grad_interaction = vec![0.0; 6 * 7 * 4];
        
        for i in 0..Y_N {
            let store_idx = X_2_DATA[i] as usize;
            let day_idx = X_1_DATA[i] as usize;
            let cat_idx = X_0_DATA[i] as usize;
            
            let mu_i = grand_mean 
                + self.store_effect_full[store_idx]
                + self.day_effect_full[day_idx]
                + self.interaction_full[store_idx * 7 * 4 + day_idx * 4 + cat_idx];
                
            let residual = Y_DATA[i] - mu_i;
            logp += log_norm_y - 0.5 * residual * residual * inv_sigma_y_sq;
            
            // Gradients
            let scaled_residual = residual * inv_sigma_y_sq;
            grad_grand_mean += scaled_residual;
            grad_store_effect[store_idx] += scaled_residual;
            grad_day_effect[day_idx] += scaled_residual;
            grad_interaction[store_idx * 7 * 4 + day_idx * 4 + cat_idx] += scaled_residual;
            
            grad_log_sigma_y += residual * residual / (sigma_y * sigma_y) - 1.0;
        }
        
        // Apply gradients
        gradient[0] += grad_grand_mean;
        gradient[123] += grad_log_sigma_y;

        // Backprop through ZeroSumTransforms
        // store_effect: 6 -> 5
        let n_store_f64 = 6.0_f64;
        let mut sum_grad_store = 0.0;
        for i in 0..5 {
            sum_grad_store += grad_store_effect[i];
        }
        let grad_fill_store = grad_store_effect[5];
        
        for i in 0..5 {
            gradient[4 + i] += grad_store_effect[i] 
                - sum_grad_store / (n_store_f64.sqrt() + n_store_f64)
                - grad_fill_store / n_store_f64.sqrt();
        }

        // day_effect: 7 -> 6  
        let n_day_f64 = 7.0_f64;
        let mut sum_grad_day = 0.0;
        for i in 0..6 {
            sum_grad_day += grad_day_effect[i];
        }
        let grad_fill_day = grad_day_effect[6];
        
        for i in 0..6 {
            gradient[9 + i] += grad_day_effect[i]
                - sum_grad_day / (n_day_f64.sqrt() + n_day_f64)
                - grad_fill_day / n_day_f64.sqrt();
        }

        // interaction: (6,7,4) -> (6,7,3) -> (6,6,3)
        // First backprop axis -1: 4->3
        for i in 0..6 {
            for j in 0..7 {
                let n_k_f64 = 4.0_f64;
                let mut sum_grad_k = 0.0;
                for k in 0..3 {
                    sum_grad_k += grad_interaction[i * 7 * 4 + j * 4 + k];
                }
                let grad_fill_k = grad_interaction[i * 7 * 4 + j * 4 + 3];
                
                for k in 0..3 {
                    self.interaction_temp[i * 7 * 3 + j * 3 + k] = 
                        grad_interaction[i * 7 * 4 + j * 4 + k]
                        - sum_grad_k / (n_k_f64.sqrt() + n_k_f64)
                        - grad_fill_k / n_k_f64.sqrt();
                }
            }
        }
        
        // Then backprop axis -2: 7->6
        for i in 0..6 {
            for k in 0..3 {
                let n_j_f64 = 7.0_f64;
                let mut sum_grad_j = 0.0;
                for j in 0..6 {
                    sum_grad_j += self.interaction_temp[i * 7 * 3 + j * 3 + k];
                }
                let grad_fill_j = self.interaction_temp[i * 7 * 3 + 6 * 3 + k];
                
                for j in 0..6 {
                    gradient[15 + i * 6 * 3 + j * 3 + k] += 
                        self.interaction_temp[i * 7 * 3 + j * 3 + k]
                        - sum_grad_j / (n_j_f64.sqrt() + n_j_f64)
                        - grad_fill_j / n_j_f64.sqrt();
                }
            }
        }

        Ok(logp)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self, _rng: &mut R, array: &[f64],
    ) -> Result<Draw, CpuMathError> {
        Ok(Draw { parameters: array.to_vec() })
    }
}