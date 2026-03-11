#![feature(autodiff)]
use std::autodiff::autodiff;
use std::time::Instant;

mod data;
mod generated;

const TOTAL: usize = 8 * 48;

#[autodiff(d_loss, Reverse, Duplicated, Active)]
fn compute_loss(input: &[f32; TOTAL]) -> f32 {
    generated::loss_fn(input, 5)
}

fn main() {
    let mut input = [0.0f32; TOTAL];
    for i in 0..TOTAL {
        input[i] = ((i as f32) * 0.01).sin();
    }

    // Warmup
    for _ in 0..500 {
        let mut grad = [0.0f32; TOTAL];
        let _ = d_loss(&input, &mut grad, 1.0);
    }

    // Benchmark forward+backward
    let n_runs = 50000;
    let start = Instant::now();
    for _ in 0..n_runs {
        let mut grad = [0.0f32; TOTAL];
        let _ = d_loss(&input, &mut grad, 1.0);
    }
    let elapsed = start.elapsed();
    let us_per_call = elapsed.as_micros() as f64 / n_runs as f64;

    println!("Enzyme minGPT forward+backward benchmark:");
    println!("  {} runs in {:.2?}", n_runs, elapsed);
    println!("  {:.1} µs/call", us_per_call);

    // Also benchmark forward-only for comparison
    let start2 = Instant::now();
    for _ in 0..n_runs {
        let _ = compute_loss(&input);
    }
    let elapsed2 = start2.elapsed();
    let us_fwd = elapsed2.as_micros() as f64 / n_runs as f64;
    println!("\nForward-only: {:.1} µs/call", us_fwd);
    println!("Backward overhead: {:.1}x", us_per_call / us_fwd);
}
