# Skill: GGUF -> PyTensor Model Translation

You are translating a GGUF model inventory into a PyTensor Python module.
The input is not source code. It is a metadata and tensor-name summary extracted
from a GGUF file. Generate code that loads weights from the GGUF file at runtime
and builds symbolic PyTensor functions.

## Output Contract

Return one complete Python module. Do not emit explanations or markdown fences.

The module must:

1. Import `numpy as np`, `pytensor`, and `pytensor.tensor as pt`.
2. Define `load_weights(gguf_path)` that loads or dequantizes GGUF tensors into `np.float32`.
3. Define pure PyTensor helpers for the model operations.
4. Define a `build_model(weights, config)` or model class that creates PyTensor graph functions.
5. Avoid embedding model weights in generated code. Use the GGUF path and tensor names.
6. Raise `NotImplementedError` for unsupported quantized tensor formats instead of silently
   treating quantized bytes as floats.
7. Include validation helpers that report missing tensors and unsupported tensor types before
   compilation.

## GGUF Input Semantics

The prompt contains:

- `path`: original GGUF file path; use it as the default model path in examples/helpers.
- `metadata`: GGUF key/value metadata. Prefer these values over hard-coded defaults.
- `tensors`: tensor names, GGUF shape order, GGML type, and file offset.

Use `gguf.GGUFReader` in generated loader code when possible:

```python
def load_weights(gguf_path):
    try:
        import gguf
    except ImportError as exc:
        raise ImportError("Install gguf to load GGUF tensors: pip install gguf") from exc

    reader = gguf.GGUFReader(gguf_path)
    weights = {}
    unsupported = {}
    for tensor in reader.tensors:
        try:
            weights[tensor.name] = materialize_tensor(tensor)
        except NotImplementedError as exc:
            unsupported[tensor.name] = str(exc)
    if unsupported:
        raise NotImplementedError(f"unsupported GGUF tensor encodings: {unsupported}")
    return weights
```

If the reader exposes a dequantized numeric array, convert it with
`np.asarray(..., dtype=np.float32)`. If it exposes quantized block bytes, call a specific
dequantization helper for that GGML type or raise `NotImplementedError`. The real
DiffusionGemma GGUF uses quantized types such as `IQ4_XS`, `IQ4_NL`, `Q5_1`, and `Q6_K`,
so a loader that only handles `F32`/`F16` must fail early.

Do not transpose blindly. GGUF tensor shapes may be stored in ggml order, and the real
DiffusionGemma inventory stores most matrices as `[input_dim, output_dim]`.

## PyTensor Patterns

Use symbolic tensor operations, not PyTorch/JAX APIs.

```python
def linear_auto(x, w, bias=None):
    input_dim = x.type.shape[-1]
    if input_dim is not None and w.get_value(borrow=True).shape[0] == input_dim:
        y = pt.dot(x, w)
    elif input_dim is not None and w.get_value(borrow=True).shape[1] == input_dim:
        y = pt.dot(x, w.T)
    else:
        # Fall back to the common DiffusionGemma GGUF orientation.
        y = pt.dot(x, w)
    if bias is not None:
        y = y + bias
    return y

def rms_norm(x, weight=None, eps=1e-6):
    y = x * pt.pow(pt.mean(pt.square(x), axis=-1, keepdims=True) + eps, -0.5)
    if weight is not None:
        y = y * weight
    return y

def gelu_pytorch_tanh(x):
    c = np.float32(0.7978845608028654)
    return 0.5 * x * (1.0 + pt.tanh(c * (x + np.float32(0.044715) * x**3)))

def softcap_logits(logits, cap):
    return pt.tanh(logits / np.float32(cap)) * np.float32(cap)
```

Use `pytensor.shared(np.asarray(weight, dtype=np.float32), name=...)` for static weights
when compiling functions. Token ids and masks should be integer symbolic variables.

For embeddings, infer orientation from `token_embd.weight`:

- If shape is `[vocab, hidden]`, use `pt.take(embedding, input_ids, axis=0)`.
- If shape is `[hidden, vocab]`, as in the downloaded DiffusionGemma GGUF, use
  `pt.take(embedding.T, input_ids, axis=0)` and use `pt.dot(hidden, embedding)` for tied
  output logits.

For top-k MoE routing, prefer `pt.sort`/`pt.argsort` and `pt.take_along_axis` if available.
If PyTensor lacks an exact op in the target environment, implement a clear NumPy-backed
`pytensor.compile.ops.as_op` fallback and isolate it behind a helper so the rest of the graph
stays symbolic.

## DiffusionGemma Target

When metadata or tensor names indicate `diffusion-gemma`, `diffusion_gemma`, `gemma4`, or
`DiffusionGemmaForBlockDiffusion`, do not generate a decoder-only causal LM. Generate a
block-diffusion forward path or a clearly scoped single-forward approximation:

- `diffusion-gemma.attention.causal` is `False` in the real GGUF inventory.
- `diffusion.canvas_length` defaults to `256`.
- Encoder/prompt tokens produce context states.
- Decoder/canvas tokens use bidirectional self-attention over the canvas plus any supported
  prompt/context attention path.
- A correct `forward_logits(input_ids, decoder_input_ids, attention_mask=None, ...)` is
  acceptable if full iterative denoising is too large. Do not substitute autoregressive
  `.generate()`.

Read config values from metadata first. For DiffusionGemma 26B-A4B, defaults are:

| Config | Metadata key | Default |
|---|---|---|
| Hidden size | `diffusion-gemma.embedding_length` | `2816` |
| Layers | `diffusion-gemma.block_count` | `30` |
| Attention heads | `diffusion-gemma.attention.head_count` | `16` |
| KV heads per layer | `diffusion-gemma.attention.head_count_kv` | `[8, ..., 2]` |
| Sliding pattern | `diffusion-gemma.attention.sliding_window_pattern` | five sliding, one full, repeated |
| Sliding window | `diffusion-gemma.attention.sliding_window` | `1024` |
| Sliding Q/K dim | `diffusion-gemma.attention.key_length_swa` | `256` |
| Full Q/K dim | `diffusion-gemma.attention.key_length` | `512` |
| Sliding V dim | `diffusion-gemma.attention.value_length_swa` | `256` |
| Full V dim | `diffusion-gemma.attention.value_length` | `512` |
| Dense FFN size | `diffusion-gemma.feed_forward_length` | `2112` |
| Expert count | `diffusion-gemma.expert_count` | `128` |
| Experts per token | `diffusion-gemma.expert_used_count` | `8` |
| Expert FFN size | `diffusion-gemma.expert_feed_forward_length` | `704` |
| RMS eps | `diffusion-gemma.attention.layer_norm_rms_epsilon` | `1e-6` |
| Vocab size | second dim of `token_embd.weight` or tokenizer metadata | `262144` |
| Logit softcap | `diffusion-gemma.final_logit_softcapping` | `30.0` |

## Real DiffusionGemma GGUF Tensor Names

Use tensor names from the inventory first. The downloaded
`diffusiongemma-26B-A4B-it-IQ4_XS.gguf` has these top-level tensors:

| Meaning | Tensor name | Shape | Type |
|---|---|---|---|
| Token embedding / tied LM head | `token_embd.weight` | `[2816, 262144]` | `Q6_K` |
| Final norm | `output_norm.weight` | `[2816]` | `F32` |
| RoPE frequencies | `rope_freqs.weight` | `[256]` | `F32` |
| Self-conditioning pre-norm | `self_cond_pre_norm.weight` | `[2816]` | `F32` |
| Self-conditioning gate | `self_cond_gate.weight` | `[2816, 2112]` | `IQ4_XS` |
| Self-conditioning up | `self_cond_up.weight` | `[2816, 2112]` | `IQ4_XS` |
| Self-conditioning down | `self_cond_down.weight` | `[2112, 2816]` | `IQ4_NL` |

There is no `output.weight` in that file. If `output.weight` is absent, treat
`token_embd.weight` as the tied LM head only after confirming its orientation.

Per-layer tensors use this pattern:

| Meaning | Tensor name |
|---|---|
| Attention input norm | `blk.{i}.attn_norm.weight` |
| Query projection | `blk.{i}.attn_q.weight` |
| Query norm | `blk.{i}.attn_q_norm.weight` |
| Key projection | `blk.{i}.attn_k.weight` |
| Key norm | `blk.{i}.attn_k_norm.weight` |
| Value projection | `blk.{i}.attn_v.weight` |
| Attention output projection | `blk.{i}.attn_output.weight` |
| Encoder output scale | `blk.{i}.enc_layer_output_scale.weight` |
| Layer output scale | `blk.{i}.layer_output_scale.weight` |
| Post-attention norm | `blk.{i}.post_attention_norm.weight` |
| Dense FFN norm | `blk.{i}.ffn_norm.weight` |
| Dense FFN gate/up/down | `blk.{i}.ffn_gate.weight`, `blk.{i}.ffn_up.weight`, `blk.{i}.ffn_down.weight` |
| Router projection | `blk.{i}.ffn_gate_inp.weight` |
| Router scale | `blk.{i}.ffn_gate_inp.scale` |
| Fused expert gate+up | `blk.{i}.ffn_gate_up_exps.weight` |
| Expert down | `blk.{i}.ffn_down_exps.weight` |
| Expert down scale | `blk.{i}.ffn_down_exps.scale` |
| Feed-forward norms | `blk.{i}.pre_ffw_norm_2.weight`, `blk.{i}.post_ffw_norm_1.weight`, `blk.{i}.post_ffw_norm_2.weight`, `blk.{i}.post_ffw_norm.weight` |

Accept optional `.weight` suffix variants with a lookup helper:

```python
def get_weight(weights, name):
    if name in weights:
        return weights[name]
    if name.endswith(".weight") and name[:-7] in weights:
        return weights[name[:-7]]
    if f"{name}.weight" in weights:
        return weights[f"{name}.weight"]
    raise KeyError(name)
```

## Attention

Use separate Q/K/V projections where present. After projection:

1. Determine per-layer head dimensions from metadata and tensor shapes.
2. Reshape Q to `[batch, seq, num_heads, head_dim]`.
3. Reshape K/V to `[batch, seq, num_kv_heads, head_dim]`.
4. Apply Q/K RMSNorm over `head_dim`.
5. Apply RoPE before attention.
6. Repeat KV heads to match Q heads.
7. Apply the correct mask for the prompt/canvas path.
8. Project attention output with `attn_output`.

For the real 26B-A4B GGUF:

- Sliding layers are `0-4, 6-10, 12-16, 18-22, 24-28`.
- Sliding layer shapes are Q `[2816, 4096]`, K `[2816, 2048]`, V `[2816, 2048]`,
  output `[4096, 2816]`, Q/K norm `[256]`.
- Full layers are `5, 11, 17, 23, 29`.
- Full layer shapes are Q `[2816, 8192]`, K `[2816, 1024]`, output `[8192, 2816]`,
  Q/K norm `[512]`.
- Full layers in this GGUF do not contain `blk.{i}.attn_v.weight`. Do not invent a value
  projection. Implement the reference behavior only if it is known; otherwise raise
  `NotImplementedError` for those layers with a clear message.

## DiffusionGemma Text Block

Implement both dense MLP and sparse MoE branch per layer:

1. `x_norm = rms_norm(x, attn_norm, eps)`.
2. Run attention and add the residual after `post_attention_norm`.
3. Dense MLP branch:
   `dense = ffn_down(gelu_pytorch_tanh(ffn_gate(ffn_norm(x))) * ffn_up(ffn_norm(x)))`.
4. MoE branch:
   - Route from the residual before dense MLP using `ffn_gate_inp.weight`.
   - Apply `ffn_gate_inp.scale` if present and shape-compatible.
   - Softmax router logits in float32.
   - Select `expert_used_count` experts.
   - Normalize selected weights to sum to 1 per token.
   - Split `ffn_gate_up_exps.weight` into gate/up halves along the 1408 dimension
     (`704 + 704`) when present.
   - Use `ffn_down_exps.weight` for expert down projections and `ffn_down_exps.scale`
     if present and shape-compatible.
5. Combine dense and MoE branches, apply the post-FFN norms present in the inventory, add
   residuals, and apply `enc_layer_output_scale.weight` or `layer_output_scale.weight`
   when present.

For DiffusionGemma RMSNorm, use the loaded norm tensor as a multiplicative scale. Do not use
Gemma 1/2's `(1 + weight)` convention unless the tensor inventory or target reference
implementation proves that this GGUF was converted with that convention.

## Self-Conditioning and Logits

DiffusionGemma accepts optional self-conditioning logits from the previous denoising step.
Implement a helper:

1. Convert previous logits to soft token embeddings with `softmax(logits) @ token_embedding.T`
   if the embedding is `[hidden, vocab]`, or `softmax(logits) @ token_embedding` if it is
   `[vocab, hidden]`.
2. Apply `self_cond_pre_norm.weight`.
3. Run the self-conditioning MLP:
   `self_cond_down(gelu_pytorch_tanh(self_cond_gate(x)) * self_cond_up(x))`.
4. Add the result to canvas token embeddings before decoder/canvas layers.

After hidden states:

```python
hidden = rms_norm(hidden, output_norm, eps)
if "output.weight" in weights:
    logits = linear_auto(hidden, output_weight)
else:
    logits = pt.dot(hidden, token_embedding)  # for [hidden, vocab] tied embeddings
logits = softcap_logits(logits.astype("float32"), final_logit_softcapping)
```

## Validation Hooks

Include these helpers even if no sample data is available:

- `list_required_tensors(config)` returns expected GGUF tensor names for the selected model path.
- `missing_tensors(weights, config)` reports absent tensors.
- `unsupported_tensors(weights_or_inventory)` reports quantized formats without dequantizers.
- `compile_forward(gguf_path)` loads weights, validates, builds the graph, and returns a compiled
  PyTensor function.

The generated code should fail early with a missing tensor list or unsupported-format list when
the GGUF inventory does not match the assumed architecture.
