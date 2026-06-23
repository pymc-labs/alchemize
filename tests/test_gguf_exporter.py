from __future__ import annotations

import struct

from click.testing import CliRunner


def test_inspect_gguf_metadata_and_tensors(tmp_path):
    from alchemize.gguf_exporter import inspect_gguf

    path = tmp_path / "gemma.gguf"
    _write_minimal_gguf(path)

    info = inspect_gguf(path)

    assert info.version == 3
    assert info.architecture == "diffusion-gemma"
    assert info.metadata["diffusion-gemma.block_count"] == 30
    assert info.metadata["diffusion-gemma.attention.causal"] is False
    assert info.metadata["diffusion.canvas_length"] == 256
    assert info.metadata["tokenizer.ggml.tokens"] == ["<pad>", "<eos>"]
    assert [tensor.name for tensor in info.tensors] == [
        "token_embd.weight",
        "self_cond_gate.weight",
        "blk.0.attn_q.weight",
        "blk.0.ffn_down_exps.scale",
        "blk.0.enc_layer_output_scale.weight",
    ]
    assert info.tensors[2].shape == [2816, 4096]
    assert info.tensors[2].type_name == "IQ4_XS"


def test_build_gguf_prompt_prioritizes_model_inventory(tmp_path):
    from alchemize.gguf_exporter import build_gguf_prompt

    path = tmp_path / "gemma.gguf"
    _write_minimal_gguf(path)

    prompt = build_gguf_prompt(path)

    assert "architecture: diffusion-gemma" in prompt
    assert "diffusion-gemma.block_count: 30" in prompt
    assert "diffusion.canvas_length: 256" in prompt
    assert prompt.index("diffusion-gemma.block_count") < prompt.index("tokenizer.ggml.tokens")
    assert prompt.index("token_embd.weight") < prompt.index("blk.0.attn_q.weight")
    assert prompt.index("self_cond_gate.weight") < prompt.index("blk.0.attn_q.weight")
    assert "token_embd.weight: shape=[2816x262144], type=Q6_K" in prompt
    assert "self_cond_gate.weight: shape=[2816x2112], type=IQ4_XS" in prompt
    assert "blk.0.attn_q.weight: shape=[2816x4096], type=IQ4_XS" in prompt
    assert "blk.0.ffn_down_exps.scale: shape=[128], type=F32" in prompt
    assert "blk.0.enc_layer_output_scale.weight: shape=[1], type=F32" in prompt


def test_cli_converts_gguf_inventory_to_pytensor(monkeypatch, tmp_path):
    from alchemize.cli import cli

    path = tmp_path / "gemma.gguf"
    _write_minimal_gguf(path)
    captured = {}

    def fake_transpile(code, source, target, *, model, verbose):
        captured["code"] = code
        captured["source"] = source
        captured["target"] = target
        return "generated pytensor code"

    monkeypatch.setattr("alchemize.cli._transpile", fake_transpile)

    result = CliRunner().invoke(cli, ["convert", str(path), "--to", "pytensor"])

    assert result.exit_code == 0
    assert result.output.strip() == "generated pytensor code"
    assert captured["source"] == "gguf"
    assert captured["target"] == "pytensor"
    assert "GGUF model inventory" in captured["code"]


def _write_minimal_gguf(path):
    with path.open("wb") as fh:
        fh.write(b"GGUF")
        fh.write(struct.pack("<IQQ", 3, 5, 5))
        _write_kv(fh, "general.architecture", 8, "diffusion-gemma")
        _write_kv(fh, "diffusion-gemma.block_count", 4, 30)
        _write_kv(fh, "diffusion-gemma.attention.causal", 7, False)
        _write_kv(fh, "diffusion.canvas_length", 4, 256)
        _write_kv(fh, "tokenizer.ggml.tokens", 9, (8, ["<pad>", "<eos>"]))

        _write_tensor_info(fh, "token_embd.weight", [2816, 262144], 14, 0)
        _write_tensor_info(fh, "self_cond_gate.weight", [2816, 2112], 23, 1024)
        _write_tensor_info(fh, "blk.0.attn_q.weight", [2816, 4096], 23, 2048)
        _write_tensor_info(fh, "blk.0.ffn_down_exps.scale", [128], 0, 3072)
        _write_tensor_info(fh, "blk.0.enc_layer_output_scale.weight", [1], 0, 4096)


def _write_kv(fh, key, value_type, value):
    _write_string(fh, key)
    fh.write(struct.pack("<I", value_type))
    if value_type == 4:
        fh.write(struct.pack("<I", value))
    elif value_type == 8:
        _write_string(fh, value)
    elif value_type == 9:
        item_type, items = value
        fh.write(struct.pack("<IQ", item_type, len(items)))
        for item in items:
            _write_string(fh, item)
    elif value_type == 7:
        fh.write(struct.pack("<?", value))
    else:
        raise AssertionError(f"unsupported test value type: {value_type}")


def _write_tensor_info(fh, name, shape, ggml_type, offset):
    _write_string(fh, name)
    fh.write(struct.pack("<I", len(shape)))
    for dim in shape:
        fh.write(struct.pack("<Q", dim))
    fh.write(struct.pack("<IQ", ggml_type, offset))


def _write_string(fh, text):
    data = text.encode("utf-8")
    fh.write(struct.pack("<Q", len(data)))
    fh.write(data)
