use crate::data::*;

// Constants for the model architecture
const N_EMBD: usize = 48;
const N_HEAD: usize = 3;
const SEQ_LEN: usize = 8;
const HEAD_DIM: usize = N_EMBD / N_HEAD; // 16
const VOCAB_SIZE: usize = 32;
const TOTAL: usize = SEQ_LEN * N_EMBD; // 384
const QKV_SIZE: usize = 3 * N_EMBD; // 144
const MLP_HIDDEN: usize = 4 * N_EMBD; // 192

// All functions use fixed-size arrays on the stack for Enzyme compatibility.
// No Vec allocations anywhere.

#[inline]
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

#[inline]
fn layer_norm_inplace(input: &[f32; N_EMBD], weight: &[f32], bias: &[f32], output: &mut [f32; N_EMBD]) {
    let mut mean = 0.0f32;
    for i in 0..N_EMBD {
        mean += input[i];
    }
    mean /= N_EMBD as f32;

    let mut var = 0.0f32;
    for i in 0..N_EMBD {
        let d = input[i] - mean;
        var += d * d;
    }
    var /= N_EMBD as f32;
    let inv_std = 1.0 / (var + 1e-5).sqrt();

    for i in 0..N_EMBD {
        output[i] = weight[i] * (input[i] - mean) * inv_std + bias[i];
    }
}

#[inline]
fn linear_embd(input: &[f32; N_EMBD], weight: &[f32], bias: &[f32], out_features: usize, output: &mut [f32]) {
    for i in 0..out_features {
        let mut sum = bias[i];
        for j in 0..N_EMBD {
            sum += input[j] * weight[i * N_EMBD + j];
        }
        output[i] = sum;
    }
}

#[inline]
fn linear_mlp_hidden(input: &[f32], weight: &[f32], bias: &[f32], in_features: usize, out_features: usize, output: &mut [f32]) {
    for i in 0..out_features {
        let mut sum = bias[i];
        for j in 0..in_features {
            sum += input[j] * weight[i * in_features + j];
        }
        output[i] = sum;
    }
}

fn causal_self_attention(
    x: &[f32; TOTAL],
    c_attn_weight: &[f32],
    c_attn_bias: &[f32],
    c_proj_weight: &[f32],
    c_proj_bias: &[f32],
    output: &mut [f32; TOTAL],
) {
    // QKV projection
    let mut q = [0.0f32; TOTAL];
    let mut k = [0.0f32; TOTAL];
    let mut v = [0.0f32; TOTAL];

    for t in 0..SEQ_LEN {
        let mut qkv = [0.0f32; QKV_SIZE];
        let input_start = t * N_EMBD;
        // Linear projection for this position
        for i in 0..QKV_SIZE {
            let mut sum = c_attn_bias[i];
            for j in 0..N_EMBD {
                sum += x[input_start + j] * c_attn_weight[i * N_EMBD + j];
            }
            qkv[i] = sum;
        }
        // Split into Q, K, V
        for c in 0..N_EMBD {
            q[t * N_EMBD + c] = qkv[c];
            k[t * N_EMBD + c] = qkv[c + N_EMBD];
            v[t * N_EMBD + c] = qkv[c + 2 * N_EMBD];
        }
    }

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Zero output
    for i in 0..TOTAL {
        output[i] = 0.0;
    }

    // Process each head
    for h in 0..N_HEAD {
        // Attention scores
        let mut att_scores = [0.0f32; SEQ_LEN * SEQ_LEN];
        for i in 0..SEQ_LEN {
            for j in 0..SEQ_LEN {
                let mut score = 0.0f32;
                for d in 0..HEAD_DIM {
                    score += q[i * N_EMBD + h * HEAD_DIM + d] * k[j * N_EMBD + h * HEAD_DIM + d];
                }
                att_scores[i * SEQ_LEN + j] = score * scale;
            }
        }

        // Causal mask + softmax per row
        for i in 0..SEQ_LEN {
            // Find max (causal: only j <= i)
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..=i {
                if att_scores[i * SEQ_LEN + j] > max_val {
                    max_val = att_scores[i * SEQ_LEN + j];
                }
            }
            // Exp and sum
            let mut sum = 0.0f32;
            for j in 0..SEQ_LEN {
                if j <= i {
                    att_scores[i * SEQ_LEN + j] = (att_scores[i * SEQ_LEN + j] - max_val).exp();
                    sum += att_scores[i * SEQ_LEN + j];
                } else {
                    att_scores[i * SEQ_LEN + j] = 0.0;
                }
            }
            // Normalize
            if sum > 0.0 {
                for j in 0..SEQ_LEN {
                    att_scores[i * SEQ_LEN + j] /= sum;
                }
            }
        }

        // Apply attention to values
        for i in 0..SEQ_LEN {
            for d in 0..HEAD_DIM {
                let mut sum = 0.0f32;
                for j in 0..SEQ_LEN {
                    sum += att_scores[i * SEQ_LEN + j] * v[j * N_EMBD + h * HEAD_DIM + d];
                }
                output[i * N_EMBD + h * HEAD_DIM + d] = sum;
            }
        }
    }

    // Output projection: apply per position, store result back
    let pre_proj = *output; // copy
    for t in 0..SEQ_LEN {
        for i in 0..N_EMBD {
            let mut sum = c_proj_bias[i];
            for j in 0..N_EMBD {
                sum += pre_proj[t * N_EMBD + j] * c_proj_weight[i * N_EMBD + j];
            }
            output[t * N_EMBD + i] = sum;
        }
    }
}

fn transformer_block(
    x: &[f32; TOTAL],
    ln_1_weight: &[f32], ln_1_bias: &[f32],
    attn_c_attn_weight: &[f32], attn_c_attn_bias: &[f32],
    attn_c_proj_weight: &[f32], attn_c_proj_bias: &[f32],
    ln_2_weight: &[f32], ln_2_bias: &[f32],
    c_fc_weight: &[f32], c_fc_bias: &[f32],
    c_proj_weight: &[f32], c_proj_bias: &[f32],
    output: &mut [f32; TOTAL],
) {
    // LayerNorm 1 + Attention
    let mut x_norm1 = [0.0f32; TOTAL];
    for t in 0..SEQ_LEN {
        let mut pos_in = [0.0f32; N_EMBD];
        let mut pos_out = [0.0f32; N_EMBD];
        for i in 0..N_EMBD { pos_in[i] = x[t * N_EMBD + i]; }
        layer_norm_inplace(&pos_in, ln_1_weight, ln_1_bias, &mut pos_out);
        for i in 0..N_EMBD { x_norm1[t * N_EMBD + i] = pos_out[i]; }
    }

    let mut attn_out = [0.0f32; TOTAL];
    causal_self_attention(&x_norm1, attn_c_attn_weight, attn_c_attn_bias, attn_c_proj_weight, attn_c_proj_bias, &mut attn_out);

    // Residual 1
    let mut x_res1 = [0.0f32; TOTAL];
    for i in 0..TOTAL { x_res1[i] = x[i] + attn_out[i]; }

    // LayerNorm 2 + MLP
    let mut x_norm2 = [0.0f32; TOTAL];
    for t in 0..SEQ_LEN {
        let mut pos_in = [0.0f32; N_EMBD];
        let mut pos_out = [0.0f32; N_EMBD];
        for i in 0..N_EMBD { pos_in[i] = x_res1[t * N_EMBD + i]; }
        layer_norm_inplace(&pos_in, ln_2_weight, ln_2_bias, &mut pos_out);
        for i in 0..N_EMBD { x_norm2[t * N_EMBD + i] = pos_out[i]; }
    }

    // MLP per position
    for t in 0..SEQ_LEN {
        let mut hidden = [0.0f32; MLP_HIDDEN];
        // FC up
        for i in 0..MLP_HIDDEN {
            let mut sum = c_fc_bias[i];
            for j in 0..N_EMBD {
                sum += x_norm2[t * N_EMBD + j] * c_fc_weight[i * N_EMBD + j];
            }
            hidden[i] = gelu(sum);
        }
        // FC down
        for i in 0..N_EMBD {
            let mut sum = c_proj_bias[i];
            for j in 0..MLP_HIDDEN {
                sum += hidden[j] * c_proj_weight[i * MLP_HIDDEN + j];
            }
            output[t * N_EMBD + i] = x_res1[t * N_EMBD + i] + sum; // residual 2
        }
    }
}

/// Forward pass: input [TOTAL] -> logits [SEQ_LEN * VOCAB_SIZE]
/// Returns logits written to output buffer.
pub fn forward(input: &[f32; TOTAL], logits: &mut [f32; SEQ_LEN * VOCAB_SIZE]) {
    let mut x = *input;

    let mut buf = [0.0f32; TOTAL];
    // Block 0
    transformer_block(&x,
        BLOCKS_0_LN_1_WEIGHT, BLOCKS_0_LN_1_BIAS,
        BLOCKS_0_ATTN_C_ATTN_WEIGHT, BLOCKS_0_ATTN_C_ATTN_BIAS,
        BLOCKS_0_ATTN_C_PROJ_WEIGHT, BLOCKS_0_ATTN_C_PROJ_BIAS,
        BLOCKS_0_LN_2_WEIGHT, BLOCKS_0_LN_2_BIAS,
        BLOCKS_0_C_FC_WEIGHT, BLOCKS_0_C_FC_BIAS,
        BLOCKS_0_C_PROJ_WEIGHT, BLOCKS_0_C_PROJ_BIAS,
        &mut buf);
    x = buf;

    // Block 1
    transformer_block(&x,
        BLOCKS_1_LN_1_WEIGHT, BLOCKS_1_LN_1_BIAS,
        BLOCKS_1_ATTN_C_ATTN_WEIGHT, BLOCKS_1_ATTN_C_ATTN_BIAS,
        BLOCKS_1_ATTN_C_PROJ_WEIGHT, BLOCKS_1_ATTN_C_PROJ_BIAS,
        BLOCKS_1_LN_2_WEIGHT, BLOCKS_1_LN_2_BIAS,
        BLOCKS_1_C_FC_WEIGHT, BLOCKS_1_C_FC_BIAS,
        BLOCKS_1_C_PROJ_WEIGHT, BLOCKS_1_C_PROJ_BIAS,
        &mut buf);
    x = buf;

    // Block 2
    transformer_block(&x,
        BLOCKS_2_LN_1_WEIGHT, BLOCKS_2_LN_1_BIAS,
        BLOCKS_2_ATTN_C_ATTN_WEIGHT, BLOCKS_2_ATTN_C_ATTN_BIAS,
        BLOCKS_2_ATTN_C_PROJ_WEIGHT, BLOCKS_2_ATTN_C_PROJ_BIAS,
        BLOCKS_2_LN_2_WEIGHT, BLOCKS_2_LN_2_BIAS,
        BLOCKS_2_C_FC_WEIGHT, BLOCKS_2_C_FC_BIAS,
        BLOCKS_2_C_PROJ_WEIGHT, BLOCKS_2_C_PROJ_BIAS,
        &mut buf);
    x = buf;

    // Final layer norm
    let mut x_final = [0.0f32; TOTAL];
    for t in 0..SEQ_LEN {
        let mut pos_in = [0.0f32; N_EMBD];
        let mut pos_out = [0.0f32; N_EMBD];
        for i in 0..N_EMBD { pos_in[i] = x[t * N_EMBD + i]; }
        layer_norm_inplace(&pos_in, LN_F_WEIGHT, LN_F_BIAS, &mut pos_out);
        for i in 0..N_EMBD { x_final[t * N_EMBD + i] = pos_out[i]; }
    }

    // LM head (no bias)
    for t in 0..SEQ_LEN {
        for i in 0..VOCAB_SIZE {
            let mut sum = 0.0f32;
            for j in 0..N_EMBD {
                sum += x_final[t * N_EMBD + j] * LM_HEAD_WEIGHT[i * N_EMBD + j];
            }
            logits[t * VOCAB_SIZE + i] = sum;
        }
    }
}

/// Cross-entropy loss for a single target position (last token).
/// input: embedded representation [TOTAL], target: target token id
/// Returns scalar loss.
pub fn loss_fn(input: &[f32; TOTAL], target: usize) -> f32 {
    let mut logits = [0.0f32; SEQ_LEN * VOCAB_SIZE];
    forward(input, &mut logits);

    // Use last position logits for loss
    let last_pos = (SEQ_LEN - 1) * VOCAB_SIZE;

    // Log-softmax of last position
    let mut max_val = f32::NEG_INFINITY;
    for i in 0..VOCAB_SIZE {
        if logits[last_pos + i] > max_val {
            max_val = logits[last_pos + i];
        }
    }

    let mut sum_exp = 0.0f32;
    for i in 0..VOCAB_SIZE {
        sum_exp += (logits[last_pos + i] - max_val).exp();
    }

    let log_sum_exp = max_val + sum_exp.ln();
    let loss = log_sum_exp - logits[last_pos + target];
    loss
}
