"""Lightweight GGUF metadata extraction for prompt-based transpilation.

The reader intentionally stops before tensor data. For Alchemize's CLI route we
only need metadata and a tensor inventory so the model can generate loader code
instead of embedding model weights in the transpiled source.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, BinaryIO

GGUF_MAGIC = b"GGUF"


class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


GGUF_TENSOR_TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
    31: "TQ1_0",
    32: "TQ2_0",
    33: "MXFP4",
}


@dataclass(frozen=True)
class GGUFTensorInfo:
    name: str
    shape: list[int]
    ggml_type: int
    offset: int

    @property
    def type_name(self) -> str:
        return GGUF_TENSOR_TYPES.get(self.ggml_type, f"TYPE_{self.ggml_type}")


@dataclass(frozen=True)
class GGUFMetadata:
    version: int
    tensor_count: int
    metadata_count: int
    metadata: dict[str, Any]
    tensors: list[GGUFTensorInfo]

    @property
    def architecture(self) -> str | None:
        value = self.metadata.get("general.architecture")
        return value if isinstance(value, str) else None


class GGUFParseError(ValueError):
    """Raised when a GGUF file cannot be parsed as metadata."""


def inspect_gguf(path: str | Path) -> GGUFMetadata:
    """Read GGUF metadata and tensor infos without loading tensor data."""
    gguf_path = Path(path)
    with gguf_path.open("rb") as fh:
        magic = fh.read(4)
        if magic != GGUF_MAGIC:
            raise GGUFParseError(f"{gguf_path} is not a GGUF file")

        version = _read_struct(fh, "<I")
        if version not in (2, 3):
            raise GGUFParseError(f"unsupported GGUF version: {version}")

        tensor_count = _read_struct(fh, "<Q")
        metadata_count = _read_struct(fh, "<Q")

        metadata = {}
        for _ in range(metadata_count):
            key = _read_string(fh)
            value_type = GGUFValueType(_read_struct(fh, "<I"))
            metadata[key] = _read_value(fh, value_type)

        tensors = []
        for _ in range(tensor_count):
            name = _read_string(fh)
            n_dims = _read_struct(fh, "<I")
            shape = [_read_struct(fh, "<Q") for _ in range(n_dims)]
            ggml_type = _read_struct(fh, "<I")
            offset = _read_struct(fh, "<Q")
            tensors.append(GGUFTensorInfo(name=name, shape=shape, ggml_type=ggml_type, offset=offset))

    return GGUFMetadata(
        version=version,
        tensor_count=tensor_count,
        metadata_count=metadata_count,
        metadata=metadata,
        tensors=tensors,
    )


def build_gguf_prompt(path: str | Path, *, max_metadata: int = 120, max_tensors: int = 400) -> str:
    """Create compact text input for GGUF -> PyTensor prompt-based transpilation."""
    gguf_path = Path(path)
    info = inspect_gguf(gguf_path)

    metadata_items = _prioritize_metadata(info.metadata)
    tensor_items = _prioritize_tensors(info.tensors)

    lines = [
        "GGUF model inventory for PyTensor transpilation.",
        f"path: {gguf_path}",
        f"gguf_version: {info.version}",
        f"architecture: {info.architecture or 'unknown'}",
        f"tensor_count: {info.tensor_count}",
        f"metadata_count: {info.metadata_count}",
        "",
        "metadata:",
    ]

    for key, value in metadata_items[:max_metadata]:
        lines.append(f"  {key}: {_format_metadata_value(value)}")
    if len(metadata_items) > max_metadata:
        lines.append(f"  ... {len(metadata_items) - max_metadata} metadata entries omitted")

    lines.extend(["", "tensors:"])
    for tensor in tensor_items[:max_tensors]:
        shape = "x".join(str(dim) for dim in tensor.shape) or "scalar"
        lines.append(f"  {tensor.name}: shape=[{shape}], type={tensor.type_name}, offset={tensor.offset}")
    if len(tensor_items) > max_tensors:
        lines.append(f"  ... {len(tensor_items) - max_tensors} tensors omitted")

    return "\n".join(lines)


def _read_struct(fh: BinaryIO, fmt: str) -> Any:
    size = struct.calcsize(fmt)
    data = fh.read(size)
    if len(data) != size:
        raise GGUFParseError("unexpected end of GGUF file")
    values = struct.unpack(fmt, data)
    return values[0] if len(values) == 1 else values


def _read_string(fh: BinaryIO) -> str:
    length = _read_struct(fh, "<Q")
    data = fh.read(length)
    if len(data) != length:
        raise GGUFParseError("unexpected end of GGUF string")
    return data.decode("utf-8")


def _read_value(fh: BinaryIO, value_type: GGUFValueType) -> Any:
    if value_type == GGUFValueType.UINT8:
        return _read_struct(fh, "<B")
    if value_type == GGUFValueType.INT8:
        return _read_struct(fh, "<b")
    if value_type == GGUFValueType.UINT16:
        return _read_struct(fh, "<H")
    if value_type == GGUFValueType.INT16:
        return _read_struct(fh, "<h")
    if value_type == GGUFValueType.UINT32:
        return _read_struct(fh, "<I")
    if value_type == GGUFValueType.INT32:
        return _read_struct(fh, "<i")
    if value_type == GGUFValueType.FLOAT32:
        return _read_struct(fh, "<f")
    if value_type == GGUFValueType.BOOL:
        return bool(_read_struct(fh, "<?"))
    if value_type == GGUFValueType.STRING:
        return _read_string(fh)
    if value_type == GGUFValueType.UINT64:
        return _read_struct(fh, "<Q")
    if value_type == GGUFValueType.INT64:
        return _read_struct(fh, "<q")
    if value_type == GGUFValueType.FLOAT64:
        return _read_struct(fh, "<d")
    if value_type == GGUFValueType.ARRAY:
        item_type = GGUFValueType(_read_struct(fh, "<I"))
        length = _read_struct(fh, "<Q")
        return [_read_value(fh, item_type) for _ in range(length)]

    raise GGUFParseError(f"unsupported GGUF value type: {value_type}")


def _prioritize_metadata(metadata: dict[str, Any]) -> list[tuple[str, Any]]:
    def sort_key(item: tuple[str, Any]) -> tuple[int, str]:
        key = item[0]
        priority = 1
        if key.startswith(("general.", "gemma", "diffusion-gemma", "diffusion_gemma", "diffusion.", "tokenizer.")):
            priority = 0
        if any(term in key for term in ("expert", "attention", "rope", "block", "context", "embedding", "canvas")):
            priority = min(priority, 0)
        return priority, key

    return sorted(metadata.items(), key=sort_key)


def _prioritize_tensors(tensors: list[GGUFTensorInfo]) -> list[GGUFTensorInfo]:
    def sort_key(tensor: GGUFTensorInfo) -> tuple[int, int, str]:
        name = tensor.name
        if name.startswith(("token_embd", "output", "rope", "self_cond")):
            return 0, -1, name
        if name.startswith("blk."):
            layer, suffix = _split_block_tensor_name(name)
            return 1, layer, suffix
        if any(term in name for term in ("attn", "ffn", "expert", "router", "layer_output_scale")):
            return 2, -1, name
        return 3, -1, name

    return sorted(tensors, key=sort_key)


def _split_block_tensor_name(name: str) -> tuple[int, str]:
    parts = name.split(".", 2)
    if len(parts) != 3:
        return 10**9, name
    try:
        layer = int(parts[1])
    except ValueError:
        layer = 10**9
    return layer, parts[2]


def _format_metadata_value(value: Any, *, max_items: int = 16, max_chars: int = 240) -> str:
    if isinstance(value, list):
        items = ", ".join(_format_metadata_value(item, max_items=max_items, max_chars=80) for item in value[:max_items])
        suffix = "" if len(value) <= max_items else f", ... ({len(value)} total)"
        text = f"[{items}{suffix}]"
    else:
        text = repr(value)

    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text
