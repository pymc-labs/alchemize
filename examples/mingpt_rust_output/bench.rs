mod generated;
mod data;

use std::time::Instant;

fn main() {
    // Create a sample input (384 = 8 * 48 floats)
    let input: Vec<f32> = (0..384).map(|i| ((i as f32) * 0.01).sin()).collect();

    // Warmup
    for _ in 0..500 {
        let _ = generated::forward(&input);
    }

    // Benchmark
    let n_runs = 50000;
    let start = Instant::now();
    for _ in 0..n_runs {
        let _ = generated::forward(&input);
    }
    let elapsed = start.elapsed();

    let us_per_call = elapsed.as_micros() as f64 / n_runs as f64;
    let ns_per_call = elapsed.as_nanos() as f64 / n_runs as f64;

    println!("Rust minGPT-nano forward pass benchmark:");
    println!("  {} runs in {:.2?}", n_runs, elapsed);
    println!("  {:.1} µs/call ({:.0} ns/call)", us_per_call, ns_per_call);

    // Also verify output is non-trivial
    let output = generated::forward(&input);
    println!("  Output length: {}", output.len());
    let sum: f32 = output.iter().sum();
    println!("  Output sum: {:.6}", sum);
}
