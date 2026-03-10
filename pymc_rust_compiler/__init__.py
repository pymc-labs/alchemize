"""PyMC Rust AI Compiler: compile PyMC models to optimized Rust via LLM."""

from pymc_rust_compiler.exporter import ModelContext, RustModelExporter, export_model
from pymc_rust_compiler.compiler import compile_model, optimize_model

__all__ = [
    "compile_model",
    "optimize_model",
    "export_model",
    "ModelContext",
    "RustModelExporter",
    "to_nutpie",
]


def to_nutpie(compile_result, model):
    """Convert a CompilationResult to a nutpie-compatible model. Lazy import."""
    from pymc_rust_compiler.nutpie_bridge import to_nutpie as _to_nutpie
    return _to_nutpie(compile_result, model)
