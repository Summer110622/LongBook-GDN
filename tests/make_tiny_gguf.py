#!/usr/bin/env python3
"""Make a small, deterministic Qwen35 GGUF for structural CPU smoke tests.

This is random test data, not a miniature trained Longbook27 checkpoint. It
exercises the all-recurrent loader and graph, including the real checkpoint's
stale full-attention interval and declared-but-absent MTP block. It says nothing
about checkpoint compatibility, output quality, quantization, or GPU support.

The tensor names and dimensions follow vnlpscale/llama.cpp at
c5709247c243a0b1ae299ddb9d8f4c8b0279be89, src/models/qwen35.cpp.
Only the Python standard library is required; weights are always generated
locally, and no model files are downloaded.
"""

from __future__ import annotations

import argparse
import math
import random
import struct
from pathlib import Path
from typing import BinaryIO, Iterable


CONTEXT_LENGTH = 256
EMBEDDING_LENGTH = 32
FEED_FORWARD_LENGTH = 64
RECURRENT_BLOCKS = 4
DECLARED_MTP_BLOCKS = 1
SSM_STATE_SIZE = 32
SSM_KEY_HEADS = 1
SSM_VALUE_HEADS = 2
SSM_INNER_SIZE = SSM_STATE_SIZE * SSM_VALUE_HEADS
SSM_CONV_KERNEL = 4
ALIGNMENT = 32

# GGUF metadata types, distinct from tensor data types.
UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY = 4, 5, 6, 7, 8, 9
GGML_TYPE_F32 = 0


def byte_tokens() -> list[str]:
    """Return GPT2's reversible byte alphabet, indexed by the original byte."""
    visible = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    extra = 256
    result = []
    for value in range(256):
        if value in visible:
            result.append(chr(value))
        else:
            result.append(chr(extra))
            extra += 1
    return result


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _value(kind: int, value: object) -> bytes:
    if kind == STRING:
        return _string(str(value))
    if kind == UINT32:
        return struct.pack("<I", value)
    if kind == INT32:
        return struct.pack("<i", value)
    if kind == FLOAT32:
        return struct.pack("<f", value)
    if kind == BOOL:
        return struct.pack("<?", value)
    if kind == ARRAY:
        item_kind, items = value
        return struct.pack("<IQ", item_kind, len(items)) + b"".join(
            _value(item_kind, item) for item in items
        )
    raise ValueError(f"unsupported GGUF metadata type: {kind}")


def _pad(stream: BinaryIO) -> None:
    stream.write(b"\0" * (-stream.tell() % ALIGNMENT))


def write_model(path: str | Path, *, seed: int = 27) -> Path:
    """Write an approximately 300 KiB F32 all-recurrent Qwen35 fixture.

    Four real blocks deliberately cross the obsolete attention interval of four.
    The fifth, declared MTP block has no tensors. No explicit recurrent-layer
    metadata is present, so the engine must infer topology from the tensors.
    """
    path = Path(path)
    rng = random.Random(seed)
    vocab = byte_tokens() + ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    metadata = [
        ("general.architecture", STRING, "qwen35"),
        ("general.name", STRING, "SYNTHETIC structural fixture; NOT Longbook27 weights"),
        ("general.description", STRING, "Deterministic random CPU smoke-test data; no trained model quality"),
        ("general.alignment", UINT32, ALIGNMENT),
        ("general.file_type", UINT32, 0),
        ("qwen35.context_length", UINT32, CONTEXT_LENGTH),
        ("qwen35.embedding_length", UINT32, EMBEDDING_LENGTH),
        ("qwen35.feed_forward_length", UINT32, FEED_FORWARD_LENGTH),
        ("qwen35.block_count", UINT32, RECURRENT_BLOCKS + DECLARED_MTP_BLOCKS),
        ("qwen35.nextn_predict_layers", UINT32, DECLARED_MTP_BLOCKS),
        ("qwen35.full_attention_interval", UINT32, 4),
        ("qwen35.attention.head_count", UINT32, 1),
        ("qwen35.attention.head_count_kv", UINT32, 1),
        ("qwen35.attention.key_length", UINT32, EMBEDDING_LENGTH),
        ("qwen35.attention.value_length", UINT32, EMBEDDING_LENGTH),
        ("qwen35.attention.layer_norm_rms_epsilon", FLOAT32, 1.0e-6),
        ("qwen35.rope.dimension_count", UINT32, EMBEDDING_LENGTH),
        ("qwen35.rope.dimension_sections", ARRAY, (INT32, [8, 8, 0, 0])),
        ("qwen35.ssm.conv_kernel", UINT32, SSM_CONV_KERNEL),
        ("qwen35.ssm.inner_size", UINT32, SSM_INNER_SIZE),
        ("qwen35.ssm.state_size", UINT32, SSM_STATE_SIZE),
        ("qwen35.ssm.time_step_rank", UINT32, SSM_VALUE_HEADS),
        ("qwen35.ssm.group_count", UINT32, SSM_KEY_HEADS),
        ("tokenizer.ggml.model", STRING, "gpt2"),
        ("tokenizer.ggml.pre", STRING, "qwen35"),
        ("tokenizer.ggml.tokens", ARRAY, (STRING, vocab)),
        ("tokenizer.ggml.token_type", ARRAY, (INT32, [1] * 256 + [3] * 3)),
        ("tokenizer.ggml.merges", ARRAY, (STRING, [])),
        ("tokenizer.ggml.bos_token_id", UINT32, 256),
        ("tokenizer.ggml.eos_token_id", UINT32, 256),
        ("tokenizer.ggml.eot_token_id", UINT32, 258),
        ("tokenizer.ggml.add_bos_token", BOOL, False),
        ("tokenizer.ggml.add_eos_token", BOOL, False),
        ("tokenizer.chat_template", STRING,
         "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + "
         "message['content'] + '<|im_end|>\\n'}}{% endfor %}"
         "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"),
    ]

    # Tensor dimensions are stored in GGML order (the contiguous dimension first).
    tensors: list[tuple[str, tuple[int, ...], bytes]] = []

    def add(name: str, shape: tuple[int, ...], values: Iterable[float] | None = None) -> None:
        count = math.prod(shape)
        floats = list(values) if values is not None else [rng.uniform(-0.05, 0.05) for _ in range(count)]
        if len(floats) != count or not all(math.isfinite(value) for value in floats):
            raise ValueError(f"invalid synthetic tensor: {name}")
        tensors.append((name, shape, struct.pack(f"<{count}f", *floats)))

    def norm(name: str, length: int) -> None:
        add(name, (length,), [1.0] * length)

    add("token_embd.weight", (EMBEDDING_LENGTH, len(vocab)))
    norm("output_norm.weight", EMBEDDING_LENGTH)

    # Opposite-paired random output rows ensure that a nonzero projection has a
    # positive printable candidate. Control/byte-only tokens have zero logits.
    # This keeps test output readable without forcing a single constant token or
    # removing the nonzero recurrent/FFN computations from the inference graph.
    output_rows = [[0.0] * EMBEDDING_LENGTH for _ in vocab]
    for token in range(33, 127, 2):
        row = [rng.uniform(-0.05, 0.05) for _ in range(EMBEDDING_LENGTH)]
        output_rows[token] = row
        output_rows[token + 1] = [-value for value in row]
    add("output.weight", (EMBEDDING_LENGTH, len(vocab)), (v for row in output_rows for v in row))

    key_dim = SSM_STATE_SIZE * SSM_KEY_HEADS
    conv_dim = 2 * key_dim + SSM_INNER_SIZE
    for index in range(RECURRENT_BLOCKS):
        prefix = f"blk.{index}."
        norm(prefix + "attn_norm.weight", EMBEDDING_LENGTH)
        norm(prefix + "post_attention_norm.weight", EMBEDDING_LENGTH)
        add(prefix + "attn_qkv.weight", (EMBEDDING_LENGTH, conv_dim))
        add(prefix + "attn_gate.weight", (EMBEDDING_LENGTH, SSM_INNER_SIZE))
        add(prefix + "ssm_conv1d.weight", (SSM_CONV_KERNEL, conv_dim))
        add(prefix + "ssm_dt.bias", (SSM_VALUE_HEADS,), [-0.5] * SSM_VALUE_HEADS)
        # Qwen35 stores negative exp(A_log), not A_log itself.
        add(prefix + "ssm_a", (SSM_VALUE_HEADS,), [-0.5] * SSM_VALUE_HEADS)
        add(prefix + "ssm_beta.weight", (EMBEDDING_LENGTH, SSM_VALUE_HEADS))
        add(prefix + "ssm_alpha.weight", (EMBEDDING_LENGTH, SSM_VALUE_HEADS))
        norm(prefix + "ssm_norm.weight", SSM_STATE_SIZE)
        add(prefix + "ssm_out.weight", (SSM_INNER_SIZE, EMBEDDING_LENGTH))
        add(prefix + "ffn_gate.weight", (EMBEDDING_LENGTH, FEED_FORWARD_LENGTH))
        add(prefix + "ffn_down.weight", (FEED_FORWARD_LENGTH, EMBEDDING_LENGTH))
        add(prefix + "ffn_up.weight", (EMBEDDING_LENGTH, FEED_FORWARD_LENGTH))

    with path.open("wb") as stream:
        stream.write(struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(metadata)))
        for key, kind, value in metadata:
            stream.write(_string(key) + struct.pack("<I", kind) + _value(kind, value))
        offset = 0
        for name, shape, data in tensors:
            stream.write(_string(name) + struct.pack("<I", len(shape)))
            stream.write(struct.pack(f"<{len(shape)}Q", *shape))
            stream.write(struct.pack("<IQ", GGML_TYPE_F32, offset))
            offset += len(data) + (-len(data) % ALIGNMENT)
        _pad(stream)
        for _name, _shape, data in tensors:
            stream.write(data)
            _pad(stream)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="path for the synthetic GGUF; parent directory must exist")
    parser.add_argument("--seed", type=int, default=27)
    args = parser.parse_args()
    path = write_model(args.output, seed=args.seed)
    print(f"Synthetic structural fixture: {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
