#!/usr/bin/env python3
"""Inspect a local GGUF v3 header and identify the pinned Longbook27 structure.

No model tensors are loaded. Identity remains unverified unless the optional
streaming SHA-256 matches the published artifact digest. Standard library only.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
from typing import BinaryIO


PROFILE_PATH = Path(__file__).resolve().parents[1] / "models" / "longbook27.json"
MAX_HEADER_BYTES = 32 * 1024 * 1024
MAX_METADATA_ITEMS = 256
MAX_TENSORS = 4096
MAX_ARRAY_ITEMS = 1_000_000
MAX_STRING_BYTES = 1024 * 1024
MAX_NAME_BYTES = 1024
SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "B", 10: "Q", 11: "q", 12: "d"}
# The pinned Q4_K_M file uses only F32, Q4_K and Q6_K tensors.
TENSOR_LAYOUTS = {0: (1, 4), 12: (256, 144), 14: (256, 210)}
SPECIAL_TOKEN_IDS = {248044, 248045, 248046}


class GGUFError(ValueError):
    """Malformed, unsupported, or unbounded GGUF metadata."""


class HeaderReader:
    def __init__(self, stream: BinaryIO, file_size: int):
        self.stream = stream
        self.file_size = file_size
        self.offset = 0
        self.descriptor_hash = None

    def read(self, size: int) -> bytes:
        if size < 0 or self.offset + size > min(self.file_size, MAX_HEADER_BYTES):
            raise GGUFError(f"Truncated header or header exceeds {MAX_HEADER_BYTES} bytes at offset {self.offset}")
        data = self.stream.read(size)
        if len(data) != size:
            raise GGUFError(f"Truncated header at offset {self.offset}")
        self.offset += size
        if self.descriptor_hash is not None:
            self.descriptor_hash.update(data)
        return data

    def scalar(self, fmt: str):
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))[0]

    def string(self, max_bytes: int = MAX_STRING_BYTES) -> str:
        length = self.scalar("Q")
        if length > max_bytes:
            raise GGUFError(f"String length {length} exceeds limit {max_bytes}")
        try:
            return self.read(length).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GGUFError("Invalid UTF-8 in GGUF string") from exc

    def value(self, value_type: int):
        if value_type == 8:
            return self.string()
        if value_type not in SCALARS:
            raise GGUFError(f"Unsupported GGUF metadata value type {value_type}")
        value = self.scalar(SCALARS[value_type])
        if value_type == 7:
            if value not in (0, 1):
                raise GGUFError("Boolean metadata must be 0 or 1")
            return bool(value)
        if isinstance(value, float) and not math.isfinite(value):
            raise GGUFError("Non-finite floating-point metadata")
        return value


def _stat_signature(st: os.stat_result) -> tuple:
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


@contextlib.contextmanager
def _open_regular(path: str | Path):
    # O_NONBLOCK lets us reject a named pipe without blocking in open().
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise GGUFError("Model path must refer to a regular file")
        stream = os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise
    with stream:
        yield stream


def inspect_gguf(path: str | Path) -> dict:
    """Read bounded metadata/descriptors and check tensor extents, never weights."""
    with _open_regular(path) as stream:
        initial_stat = os.fstat(stream.fileno())
        if not stat.S_ISREG(initial_stat.st_mode):
            raise GGUFError("Model path must refer to a regular file")
        reader = HeaderReader(stream, initial_stat.st_size)
        if reader.read(4) != b"GGUF":
            raise GGUFError("Expected little-endian GGUF magic")
        version = reader.scalar("I")
        if version != 3:
            raise GGUFError(f"Only GGUF version 3 is supported, found {version}")
        tensor_count, metadata_count = reader.scalar("Q"), reader.scalar("Q")
        if not 0 < tensor_count <= MAX_TENSORS:
            raise GGUFError(f"Invalid or excessive tensor count {tensor_count}")
        if not 0 < metadata_count <= MAX_METADATA_ITEMS:
            raise GGUFError(f"Invalid or excessive metadata count {metadata_count}")
        metadata, arrays, special_tokens = {}, {}, {}
        for _ in range(metadata_count):
            key = reader.string(MAX_NAME_BYTES)
            if not key or "\x00" in key or key in metadata:
                raise GGUFError(f"Empty, duplicate or invalid metadata key {key!r}")
            value_type = reader.scalar("I")
            if value_type != 9:
                metadata[key] = reader.value(value_type)
                continue
            element_type, length = reader.scalar("I"), reader.scalar("Q")
            if element_type not in SCALARS and element_type != 8:
                raise GGUFError(f"Invalid array element type {element_type}; nested arrays are unsupported")
            if length > MAX_ARRAY_ITEMS:
                raise GGUFError(f"Array length {length} exceeds limit {MAX_ARRAY_ITEMS}")
            arrays[key] = {"element_type": element_type, "length": length}
            values = []
            # Stream each value so the tokenizer never becomes a large Python list.
            for index in range(length):
                value = reader.value(element_type)
                if length <= 64:
                    values.append(value)
                if key == "tokenizer.ggml.tokens" and index in SPECIAL_TOKEN_IDS:
                    special_tokens[str(index)] = value
            metadata[key] = values if length <= 64 else {"array": arrays[key]}
        metadata_end = reader.offset
        alignment = metadata.get("general.alignment", 32)
        if type(alignment) is not int or not 1 <= alignment <= 4096 or alignment & (alignment - 1):
            raise GGUFError("GGUF alignment must be a power of two from 1 through 4096")
        tensors = {}
        reader.descriptor_hash = hashlib.sha256()
        for _ in range(tensor_count):
            name = reader.string(MAX_NAME_BYTES)
            if not name or "\x00" in name or name in tensors:
                raise GGUFError(f"Empty, duplicate or invalid tensor name {name!r}")
            dimension_count = reader.scalar("I")
            if not 1 <= dimension_count <= 4:
                raise GGUFError(f"Invalid dimension count for tensor {name}")
            dimensions = [reader.scalar("Q") for _ in range(dimension_count)]
            if any(d <= 0 or d > (1 << 31) for d in dimensions):
                raise GGUFError(f"Invalid dimensions for tensor {name}")
            ggml_type, offset = reader.scalar("I"), reader.scalar("Q")
            if ggml_type not in TENSOR_LAYOUTS:
                raise GGUFError(f"Unsupported tensor type {ggml_type} for {name}; this inspector supports F32/Q4_K/Q6_K")
            block_size, block_bytes = TENSOR_LAYOUTS[ggml_type]
            if dimensions[0] % block_size:
                raise GGUFError(f"Quantized row is not a multiple of {block_size} for {name}")
            if offset % alignment:
                raise GGUFError(f"Misaligned tensor offset for {name}")
            byte_count = math.prod(dimensions) // block_size * block_bytes
            tensors[name] = {"dimensions": dimensions, "ggml_type": ggml_type, "offset": offset, "size_bytes": byte_count}
        descriptor_end = reader.offset
        data_offset = (descriptor_end + alignment - 1) // alignment * alignment
        if data_offset > initial_stat.st_size:
            raise GGUFError("Missing tensor data section")
        last_end = 0
        for name, tensor in sorted(tensors.items(), key=lambda item: item[1]["offset"]):
            end = tensor["offset"] + tensor["size_bytes"]
            if tensor["offset"] < last_end:
                raise GGUFError(f"Overlapping tensor extent for {name}")
            if end > initial_stat.st_size - data_offset:
                raise GGUFError(f"Tensor {name} extends beyond the file")
            last_end = end
        if _stat_signature(initial_stat) != _stat_signature(os.fstat(stream.fileno())):
            raise GGUFError("Model file changed during inspection")
        return {
            "file_size": initial_stat.st_size,
            "version": version,
            "metadata_count": metadata_count,
            "tensor_count": tensor_count,
            "metadata": metadata,
            "arrays": arrays,
            "special_tokens": special_tokens,
            "tensors": tensors,
            "metadata_end_offset": metadata_end,
            "tensor_table_end_offset": descriptor_end,
            "tensor_data_offset": data_offset,
            "tensor_descriptor_sha256": reader.descriptor_hash.hexdigest(),
            "stat_signature": _stat_signature(initial_stat),
        }


def load_profile() -> dict:
    with PROFILE_PATH.open(encoding="utf-8") as stream:
        return json.load(stream)


def expected_tensors(profile: dict) -> dict:
    topology = profile["topology"]
    result = {name: dict(spec) for name, spec in topology["global_tensors"].items()}
    for layer in range(topology["main_recurrent_layers"]):
        for suffix, spec in topology["block_tensors"].items():
            result[f"blk.{layer}.{suffix}"] = {
                "dimensions": spec["dimensions"],
                "ggml_type": 14 if layer in spec.get("q6_layers", []) else spec["ggml_type"],
            }
    return result


def _equal(actual, expected) -> bool:
    if isinstance(expected, float):
        return type(actual) in (float, int) and math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-12)
    return type(actual) is type(expected) and actual == expected


def validate_target(info: dict, profile: dict) -> list[str]:
    """Check the artifact's structure. Matching these checks is not weight identity."""
    errors = []
    expected = profile["gguf"]
    if info["file_size"] != profile["artifact"]["size_bytes"]:
        errors.append(f"File size {info['file_size']} does not match {profile['artifact']['size_bytes']} bytes")
    for key in ("version", "metadata_count", "tensor_count", "metadata_end_offset", "tensor_table_end_offset", "tensor_data_offset", "tensor_descriptor_sha256"):
        if info[key] != expected[key]:
            errors.append(f"GGUF {key} does not match the pinned artifact")
    for key, value in expected["metadata"].items():
        if not _equal(info["metadata"].get(key), value):
            errors.append(f"Metadata {key} must equal {value!r}")
    for key in expected["absent_metadata"]:
        if key in info["metadata"]:
            errors.append(f"Unexpected metadata {key} for this pinned artifact")
    for key, value in expected["arrays"].items():
        if info["arrays"].get(key) != value:
            errors.append(f"Tokenizer array {key} has unexpected type or length")
    if info["special_tokens"] != expected["special_tokens"]:
        errors.append("Special token IDs/content do not match the target tokenizer")
    template = info["metadata"].get("tokenizer.chat_template")
    if not isinstance(template, str) or hashlib.sha256(template.encode()).hexdigest() != expected["chat_template_sha256"]:
        errors.append("Embedded chat template does not match the pinned artifact")
    tensor_specs = expected_tensors(profile)
    missing = set(tensor_specs) - set(info["tensors"])
    extra = set(info["tensors"]) - set(tensor_specs)
    if missing:
        errors.append("Missing required tensors: " + ", ".join(sorted(missing)[:8]))
    if extra:
        errors.append("Unexpected tensors (including unsupported MTP/vision): " + ", ".join(sorted(extra)[:8]))
    for name, spec in tensor_specs.items():
        tensor = info["tensors"].get(name)
        if tensor is not None and any(tensor[key] != spec[key] for key in ("dimensions", "ggml_type")):
            errors.append(f"Tensor dimensions or quantization mismatch: {name}")
    counts = {str(k): v for k, v in collections.Counter(t["ggml_type"] for t in info["tensors"].values()).items()}
    if counts != expected["tensor_type_counts"]:
        errors.append("Tensor quantization type counts do not match the pinned artifact")
    return errors


def hash_file(path: str | Path, expected_stat: tuple | None = None) -> str:
    """Hash with bounded memory; this explicitly reads the full weight file."""
    digest = hashlib.sha256()
    with _open_regular(path) as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise GGUFError("Model path must refer to a regular file")
        signature = _stat_signature(before)
        if expected_stat is not None and signature != expected_stat:
            raise GGUFError("Model file changed between inspection and checksum")
        while True:
            chunk = stream.read(4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        if signature != _stat_signature(os.fstat(stream.fileno())):
            raise GGUFError("Model file changed during checksum")
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Local GGUF path; no download is performed")
    parser.add_argument("--verify-sha256", action="store_true", help="Stream the complete file and compare its published SHA-256")
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable report")
    args = parser.parse_args(argv)
    try:
        profile = load_profile()
        info = inspect_gguf(args.model)
        errors = validate_target(info, profile)
        report = {
            "model": str(args.model),
            "profile": profile["profile_id"],
            "structural_match": not errors,
            "checkpoint_identity": "unverified",
            "sha256": {"status": "not_checked"},
            "file_size_bytes": info["file_size"],
            "header_and_descriptor_bytes": info["tensor_table_end_offset"],
            "weight_tensors_loaded": False,
            "gguf_version": info["version"],
            "architecture": info["metadata"].get("general.architecture"),
            "declared_blocks": info["metadata"].get("qwen35.block_count"),
            "declared_mtp_blocks": info["metadata"].get("qwen35.nextn_predict_layers"),
            "recurrent_tensor_layers": sum(name.endswith(".ssm_conv1d.weight") for name in info["tensors"]),
            "tensor_count": info["tensor_count"],
            "text_only_client": True,
            "errors": errors,
        }
        if args.verify_sha256 and not errors:
            print("Computing full-file SHA-256...", file=sys.stderr)
            digest = hash_file(args.model, info["stat_signature"])
            matched = digest == profile["artifact"]["sha256"]
            report["sha256"] = {"status": "matched" if matched else "mismatch", "computed": digest}
            report["checkpoint_identity"] = "matches_pinned_sha256" if matched else "mismatch"
            if not matched:
                errors.append("Full-file SHA-256 does not match the published artifact")
        elif args.verify_sha256:
            report["sha256"] = {"status": "not_checked", "reason": "structure_mismatch"}
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(f"Longbook27 structure: {'MATCH' if report['structural_match'] else 'MISMATCH'}")
            print(f"GGUF v{info['version']}; {info['tensor_count']} tensor descriptors; {info['file_size']:,} bytes")
            print(f"Declared blocks: {report['declared_blocks']}; recurrent tensor layers: {report['recurrent_tensor_layers']}; declared MTP: {report['declared_mtp_blocks']}")
            print(f"Checkpoint identity: {report['checkpoint_identity']}")
            if report["sha256"]["status"] == "not_checked":
                print("SHA-256 not checked. Matching metadata does not establish weight identity.")
            else:
                print(f"SHA-256: {report['sha256']['computed']} ({report['sha256']['status']})")
            print("Client modality: text. No compatible vision projector is bundled or verified.")
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 2 if errors else 0
    except (GGUFError, OSError, ValueError, KeyError) as exc:
        if args.json:
            print(json.dumps({"structural_match": False, "checkpoint_identity": "unverified", "errors": [str(exc)]}))
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
