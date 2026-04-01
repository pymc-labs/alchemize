"""Transalchemy: compile and transpile models across frameworks via LLM.

Supports: PyMC/Stan → Rust, Stan → PyMC, JAX ↔ PyTorch.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from alchemize.jax_exporter import (
    JaxModelExporter,
    export_jax_model,
)

# JAX ↔ PyTorch (no heavy deps at import time)
from alchemize.jax_pytorch_transpiler import (
    TranspileResult,
    transpile_jax_to_pytorch,
    transpile_pytorch_to_jax,
)
from alchemize.pytorch_exporter import (
    PytorchModelExporter,
    export_pytorch_model,
)
from alchemize.pytorch_rust_transpiler import (
    RustTranspileResult,
    transpile_pytorch_to_rust,
)

# PyMC/Stan imports are lazy — they pull in heavy deps (pymc, bridgestan)
if TYPE_CHECKING:
    from alchemize.analysis import (
        plot_optimization_progress,
        plot_timeline,
        plot_waterfall,
        print_summary,
    )
    from alchemize.compiler import (
        OptimizationEvent,
        compile_model,
        optimize_model,
    )
    from alchemize.exporter import (
        ModelContext,
        RustModelExporter,
        export_model,
    )
    from alchemize.stan_compiler import (
        StanCompilationResult,
        compile_stan_model,
    )
    from alchemize.stan_exporter import (
        StanModelContext,
        StanModelExporter,
        export_stan_model,
    )
    from alchemize.stan_to_pymc import StanToPyMCResult, transpile_stan_to_pymc


def __getattr__(name: str):
    """Lazy import for PyMC/Stan components."""
    _lazy_imports = {
        "ModelContext": ("alchemize.exporter", "ModelContext"),
        "RustModelExporter": ("alchemize.exporter", "RustModelExporter"),
        "export_model": ("alchemize.exporter", "export_model"),
        "compile_model": ("alchemize.compiler", "compile_model"),
        "optimize_model": ("alchemize.compiler", "optimize_model"),
        "OptimizationEvent": ("alchemize.compiler", "OptimizationEvent"),
        "plot_optimization_progress": (
            "alchemize.analysis",
            "plot_optimization_progress",
        ),
        "plot_waterfall": ("alchemize.analysis", "plot_waterfall"),
        "plot_timeline": ("alchemize.analysis", "plot_timeline"),
        "print_summary": ("alchemize.analysis", "print_summary"),
        "StanModelContext": ("alchemize.stan_exporter", "StanModelContext"),
        "StanModelExporter": ("alchemize.stan_exporter", "StanModelExporter"),
        "export_stan_model": ("alchemize.stan_exporter", "export_stan_model"),
        "compile_stan_model": (
            "alchemize.stan_compiler",
            "compile_stan_model",
        ),
        "StanCompilationResult": (
            "alchemize.stan_compiler",
            "StanCompilationResult",
        ),
        "transpile_stan_to_pymc": (
            "alchemize.stan_to_pymc",
            "transpile_stan_to_pymc",
        ),
        "StanToPyMCResult": ("alchemize.stan_to_pymc", "StanToPyMCResult"),
    }
    if name in _lazy_imports:
        module_path, attr = _lazy_imports[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module 'alchemize' has no attribute {name!r}")


__all__ = [
    # PyMC → Rust (lazy)
    "compile_model",
    "optimize_model",
    "OptimizationEvent",
    # Analysis / plotting
    "plot_optimization_progress",
    "plot_waterfall",
    "plot_timeline",
    "print_summary",
    "export_model",
    "ModelContext",
    "RustModelExporter",
    "to_nutpie",
    # Stan → Rust (lazy)
    "compile_stan_model",
    "export_stan_model",
    "StanModelContext",
    "StanModelExporter",
    "StanCompilationResult",
    # Stan → PyMC (lazy)
    "transpile_stan_to_pymc",
    "StanToPyMCResult",
    # JAX ↔ PyTorch
    "transpile_jax_to_pytorch",
    "transpile_pytorch_to_jax",
    "TranspileResult",
    "JaxModelExporter",
    "export_jax_model",
    "PytorchModelExporter",
    "export_pytorch_model",
    # PyTorch → Rust
    "transpile_pytorch_to_rust",
    "RustTranspileResult",
]


def to_nutpie(compile_result, model):
    """Convert a CompilationResult to a nutpie-compatible model. Lazy import."""
    from alchemize.nutpie_bridge import to_nutpie as _to_nutpie

    return _to_nutpie(compile_result, model)
