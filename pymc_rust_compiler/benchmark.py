"""Benchmark: compare PyMC (nutpie) vs AI-compiled Rust sampler."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm


def benchmark_nutpie(model: pm.Model, draws: int = 2000, tune: int = 1000, chains: int = 4) -> dict:
    """Benchmark PyMC sampling with nutpie backend."""
    print(f"  nutpie: {chains} chains x {draws} draws...")
    start = time.time()
    idata = pm.sample(
        draws=draws,
        tune=tune,
        chains=chains,
        nuts_sampler="nutpie",
        model=model,
        random_seed=42,
        progressbar=False,
    )
    elapsed = time.time() - start
    throughput = (chains * draws) / elapsed

    return {
        "backend": "nutpie",
        "elapsed_s": elapsed,
        "throughput": throughput,
        "idata": idata,
    }


def benchmark_rust(build_dir: str | Path, draws: int = 2000, tune: int = 1000, chains: int = 4) -> dict:
    """Benchmark the AI-compiled Rust sampler."""
    build_dir = Path(build_dir)
    binary = build_dir / "target" / "release" / "sample"

    if not binary.exists():
        # Build with the sampler main
        print("  Building Rust sampler...")
        subprocess.run(
            ["cargo", "build", "--release", "--bin", "sample"],
            cwd=build_dir,
            capture_output=True,
            check=True,
        )

    print(f"  Rust: {chains} chains x {draws} draws...")
    start = time.time()
    result = subprocess.run(
        [str(binary)],
        cwd=build_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:500]}")
        return {"backend": "rust", "elapsed_s": elapsed, "error": result.stderr}

    throughput = (chains * draws) / elapsed

    return {
        "backend": "rust-ai",
        "elapsed_s": elapsed,
        "throughput": throughput,
        "output": result.stdout,
    }


# ---------------------------------------------------------------------------
# logp+dlogp evaluation benchmark (no sampling overhead)
# ---------------------------------------------------------------------------

def benchmark_logp_pytensor(model: pm.Model, n_evals: int = 100_000) -> dict:
    """Benchmark PyTensor's compiled logp+dlogp function (what nutpie calls)."""
    logp_dlogp_fn = model.logp_dlogp_function(ravel_inputs=True)
    logp_dlogp_fn.set_extra_values({})

    # Build unconstrained parameter vector from initial point
    ip = model.initial_point()
    x0 = np.concatenate([np.atleast_1d(ip[v.name]) for v in logp_dlogp_fn._grad_vars])

    # Warmup
    for _ in range(200):
        logp_dlogp_fn(x0)

    # Timed run
    start = time.perf_counter()
    for _ in range(n_evals):
        logp_val, dlogp_val = logp_dlogp_fn(x0)
    elapsed = time.perf_counter() - start

    us_per_eval = (elapsed / n_evals) * 1e6

    return {
        "backend": "pytensor",
        "n_evals": n_evals,
        "elapsed_s": elapsed,
        "us_per_eval": us_per_eval,
        "logp": float(logp_val),
    }


def benchmark_logp_rust(build_dir: str | Path, model: pm.Model, n_evals: int = 100_000) -> dict:
    """Benchmark the AI-compiled Rust logp+dlogp function."""
    build_dir = Path(build_dir).resolve()
    binary = build_dir / "target" / "release" / "bench"

    if not binary.exists():
        print("  Building Rust bench binary...")
        result = subprocess.run(
            ["cargo", "build", "--release", "--bin", "bench"],
            cwd=build_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return {"backend": "rust-ai", "error": f"Build failed: {result.stderr[:500]}"}

    # Get initial unconstrained point
    logp_dlogp_fn = model.logp_dlogp_function(ravel_inputs=True)
    logp_dlogp_fn.set_extra_values({})
    ip = model.initial_point()
    x0 = np.concatenate([np.atleast_1d(ip[v.name]) for v in logp_dlogp_fn._grad_vars])

    # Prepare stdin: first line = n_iters, second line = param vector
    param_str = ",".join(f"{v:.17e}" for v in x0)
    stdin_data = f"{n_evals}\n{param_str}\n"

    result = subprocess.run(
        [str(binary)],
        cwd=build_dir,
        input=stdin_data,
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        return {"backend": "rust-ai", "error": f"Run failed: {result.stderr[:500]}"}

    # Parse output: us_per_eval,logp
    parts = result.stdout.strip().split(",")
    us_per_eval = float(parts[0])
    logp = float(parts[1])

    elapsed = us_per_eval * n_evals / 1e6

    return {
        "backend": "rust-ai",
        "n_evals": n_evals,
        "elapsed_s": elapsed,
        "us_per_eval": us_per_eval,
        "logp": logp,
    }


def print_logp_comparison(pytensor_result: dict, rust_result: dict, model_name: str = ""):
    """Print logp+dlogp evaluation benchmark comparison."""
    header = f"LOGP+DLOGP BENCHMARK{f': {model_name}' if model_name else ''}"
    print("\n" + "=" * 65)
    print(header)
    print("=" * 65)

    print(f"\n{'Backend':<20} {'us/eval':<12} {'evals/sec':<14} {'Speedup':<10}")
    print("-" * 56)

    pt = pytensor_result
    if "error" in pt:
        print(f"{'pytensor':<20} {'ERROR':<12}")
    else:
        evals_per_sec_pt = 1e6 / pt["us_per_eval"]
        print(f"{'pytensor':<20} {pt['us_per_eval']:<12.2f} {evals_per_sec_pt:<14,.0f} {'1.00x':<10}")

    rs = rust_result
    if "error" in rs:
        print(f"{'rust-ai':<20} {'ERROR':<12}  {rs['error'][:60]}")
    else:
        evals_per_sec_rs = 1e6 / rs["us_per_eval"]
        speedup = pt["us_per_eval"] / rs["us_per_eval"] if "error" not in pt else 0
        print(f"{'rust-ai':<20} {rs['us_per_eval']:<12.2f} {evals_per_sec_rs:<14,.0f} {speedup:<10.2f}x")

        # Check logp agreement
        if "error" not in pt:
            logp_diff = abs(pt["logp"] - rs["logp"])
            rel_err = logp_diff / max(abs(pt["logp"]), 1e-10)
            status = "MATCH" if rel_err < 1e-4 else f"MISMATCH (rel_err={rel_err:.2e})"
            print(f"\n  logp check: pytensor={pt['logp']:.8f}  rust={rs['logp']:.8f}  [{status}]")

    print()


def print_comparison(nutpie_result: dict, rust_result: dict):
    """Print a nice comparison table."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print(f"\n{'Backend':<20} {'Time (s)':<12} {'Draws/sec':<12} {'Speedup':<10}")
    print("-" * 54)

    nt = nutpie_result["elapsed_s"]
    print(f"{'nutpie':<20} {nt:<12.2f} {nutpie_result['throughput']:<12.0f} {'1.00x':<10}")

    if "error" not in rust_result:
        rt = rust_result["elapsed_s"]
        speedup = nt / rt
        print(f"{'rust-ai':<20} {rt:<12.2f} {rust_result['throughput']:<12.0f} {speedup:<10.2f}x")
    else:
        print(f"{'rust-ai':<20} {'FAILED':<12}")

    print()
