"""Benchmark logp+dlogp evaluation: PyTensor vs AI-compiled Rust.

This measures raw logp+gradient computation speed without any sampling overhead.
It's the fairest comparison of what the Rust compiler actually produces vs
what nutpie calls under the hood (pytensor's compiled logp_dlogp_function).

Usage:
    cd pymc-rust-ai-compiler
    uv run python examples/bench_logp.py
"""

import numpy as np
import pymc as pm

from pymc_rust_compiler.benchmark import (
    benchmark_logp_pytensor,
    benchmark_logp_rust,
    print_logp_comparison,
)

N_EVALS = 500_000


def make_normal_model():
    """Simple 2-parameter model."""
    np.random.seed(42)
    y_obs = np.random.normal(3.0, 1.5, size=100)
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=5)
        pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)
    return model, "compiled_models/normal"


def make_linreg_model():
    """Linear regression, 3 parameters."""
    np.random.seed(42)
    N = 200
    x = np.random.randn(N)
    y_obs = 2.5 + 1.3 * x + np.random.normal(0, 0.8, N)
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=5)
        mu = alpha + beta * x
        pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)
    return model, "compiled_models/linreg"


def make_hierarchical_model():
    """Hierarchical model, 12 unconstrained parameters."""
    np.random.seed(42)
    n_groups = 8
    n_per_group = np.random.randint(10, 30, size=n_groups)
    N = n_per_group.sum()
    true_a = np.random.normal(1.5, 0.7, n_groups)
    group_idx = np.repeat(np.arange(n_groups), n_per_group)
    x = np.random.binomial(1, 0.5, N).astype(float)
    y_obs = true_a[group_idx] + -0.8 * x + np.random.normal(0, 0.5, N)
    with pm.Model() as model:
        mu_a = pm.Normal("mu_a", mu=0, sigma=10)
        sigma_a = pm.HalfNormal("sigma_a", sigma=5)
        a_offset = pm.Normal("a_offset", mu=0, sigma=1, shape=n_groups)
        a = pm.Deterministic("a", mu_a + sigma_a * a_offset)
        b = pm.Normal("b", mu=0, sigma=10)
        sigma_y = pm.HalfNormal("sigma_y", sigma=5)
        mu_y = a[group_idx] + b * x
        pm.Normal("y", mu=mu_y, sigma=sigma_y, observed=y_obs)
    return model, "compiled_models/hierarchical"


def main():
    models = [
        ("Normal (2 params)", make_normal_model),
        ("LinReg (3 params)", make_linreg_model),
        ("Hierarchical (12 params)", make_hierarchical_model),
    ]

    results = []
    for name, make_fn in models:
        print(f"\n{'='*65}")
        print(f"  {name}")
        print(f"{'='*65}")

        model, build_dir = make_fn()
        n_evals = N_EVALS

        print(f"  Running {n_evals:,} logp+dlogp evaluations...")

        pt_result = benchmark_logp_pytensor(model, n_evals=n_evals)
        print(f"    pytensor: {pt_result['us_per_eval']:.2f} us/eval")

        rs_result = benchmark_logp_rust(build_dir, model, n_evals=n_evals)
        if "error" in rs_result:
            print(f"    rust-ai: ERROR - {rs_result['error'][:100]}")
        else:
            print(f"    rust-ai:  {rs_result['us_per_eval']:.2f} us/eval")

        print_logp_comparison(pt_result, rs_result, model_name=name)
        results.append((name, pt_result, rs_result))

    # Summary table
    print("\n" + "=" * 65)
    print("SUMMARY: logp+dlogp evaluation speed")
    print("=" * 65)
    print(f"\n{'Model':<30} {'pytensor':<14} {'rust-ai':<14} {'Speedup':<10}")
    print("-" * 68)
    for name, pt, rs in results:
        pt_us = f"{pt['us_per_eval']:.2f} us" if "error" not in pt else "ERROR"
        if "error" not in rs:
            rs_us = f"{rs['us_per_eval']:.2f} us"
            speedup = f"{pt['us_per_eval'] / rs['us_per_eval']:.1f}x" if "error" not in pt else "?"
        else:
            rs_us = "ERROR"
            speedup = "-"
        print(f"  {name:<28} {pt_us:<14} {rs_us:<14} {speedup:<10}")
    print()


if __name__ == "__main__":
    main()
