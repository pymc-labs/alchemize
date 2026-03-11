#![feature(autodiff)]
use std::autodiff::autodiff;

mod data;
mod generated;

const TOTAL: usize = 8 * 48; // SEQ_LEN * N_EMBD

/// Wrapper that Enzyme can differentiate: takes input array, returns scalar loss.
/// Target token is hardcoded (Enzyme needs pure f32 in/out).
#[autodiff(d_loss, Reverse, Duplicated, Active)]
fn compute_loss(input: &[f32; TOTAL]) -> f32 {
    generated::loss_fn(input, 5) // target token = 5
}

fn main() {
    // Create sample input
    let mut input = [0.0f32; TOTAL];
    for i in 0..TOTAL {
        input[i] = ((i as f32) * 0.01).sin();
    }

    // Forward pass: compute loss
    let loss = compute_loss(&input);
    println!("Loss = {:.6}", loss);

    // Compute gradients via Enzyme reverse-mode AD
    let mut grad_input = [0.0f32; TOTAL];
    let loss_val = d_loss(&input, &mut grad_input, 1.0);

    println!("Loss (from AD) = {:.6}", loss_val);
    println!("Gradient shape: {} elements", grad_input.len());

    // Print some gradient statistics
    let grad_sum: f32 = grad_input.iter().sum();
    let grad_abs_sum: f32 = grad_input.iter().map(|x| x.abs()).sum();
    let grad_max: f32 = grad_input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let grad_min: f32 = grad_input.iter().cloned().fold(f32::INFINITY, f32::min);

    println!("Gradient stats:");
    println!("  sum     = {:.6}", grad_sum);
    println!("  abs_sum = {:.6}", grad_abs_sum);
    println!("  max     = {:.6}", grad_max);
    println!("  min     = {:.6}", grad_min);

    // Print first 10 gradient values
    println!("First 10 gradients:");
    for i in 0..10 {
        println!("  grad[{}] = {:.8}", i, grad_input[i]);
    }

    // Verify gradient is non-trivial (not all zeros)
    let non_zero = grad_input.iter().filter(|&&x| x.abs() > 1e-10).count();
    println!("Non-zero gradients: {} / {}", non_zero, TOTAL);

    if non_zero > 0 {
        println!("\nEnzyme reverse-mode AD on minGPT: SUCCESS!");

        // Numerical gradient check for first element
        let eps = 1e-3;
        let mut input_plus = input;
        input_plus[0] += eps;
        let loss_plus = compute_loss(&input_plus);

        let mut input_minus = input;
        input_minus[0] -= eps;
        let loss_minus = compute_loss(&input_minus);

        let numerical_grad = (loss_plus - loss_minus) / (2.0 * eps);
        let enzyme_grad = grad_input[0];
        let rel_error = (enzyme_grad - numerical_grad).abs() / (numerical_grad.abs() + 1e-8);

        println!("\nGradient check (element 0):");
        println!("  Enzyme:    {:.8}", enzyme_grad);
        println!("  Numerical: {:.8}", numerical_grad);
        println!("  Rel error: {:.2e}", rel_error);
    } else {
        println!("\nWARNING: All gradients are zero!");
    }
}
