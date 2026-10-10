"""Synthetic format/guard checks; these never validate real model inference."""

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "model_info.py"
SPEC = importlib.util.spec_from_file_location("model_info", MODULE_PATH)
model_info = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model_info)


def text_bytes(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def item(key, kind, value):
    prefix = text_bytes(key) + struct.pack("<I", kind)
    if kind == 8:
        return prefix + text_bytes(value)
    return prefix + struct.pack("<" + model_info.SCALARS[kind], value)


def tiny_gguf(metadata=None, tensors=None, payload_size=64):
    if metadata is None:
        metadata = [item("general.architecture", 8, "qwen35")]
    if tensors is None:
        tensors = [("synthetic.weight", [4], 0, 0)]
    result = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata)) + b"".join(metadata)
    for name, dimensions, kind, offset in tensors:
        result += text_bytes(name) + struct.pack("<I", len(dimensions))
        result += struct.pack("<" + "Q" * len(dimensions), *dimensions)
        result += struct.pack("<IQ", kind, offset)
    result += bytes((-len(result)) % 32)
    return result + b"X" * payload_size


class InspectorFormatTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "synthetic.gguf"

    def inspect(self, data):
        self.path.write_bytes(data)
        return model_info.inspect_gguf(self.path)

    def test_valid_synthetic_header_is_not_the_target_model(self):
        info = self.inspect(tiny_gguf())
        self.assertEqual(info["metadata"]["general.architecture"], "qwen35")
        self.assertEqual(info["tensors"]["synthetic.weight"]["size_bytes"], 16)
        self.assertLess(info["tensor_table_end_offset"], info["file_size"])
        errors = model_info.validate_target(info, model_info.load_profile())
        self.assertTrue(any("Missing required tensors" in error for error in errors))
        self.assertTrue(any("File size" in error for error in errors))

    def test_reader_does_not_consume_tensor_payload(self):
        positions = []
        original = model_info.HeaderReader.read

        def tracked(reader, count):
            value = original(reader, count)
            positions.append(reader.offset)
            return value

        with mock.patch.object(model_info.HeaderReader, "read", tracked):
            info = self.inspect(tiny_gguf(payload_size=4096))
        self.assertEqual(max(positions), info["tensor_table_end_offset"])
        self.assertLessEqual(max(positions), info["tensor_data_offset"])

    def test_invalid_magic_version_and_truncation(self):
        data = tiny_gguf()
        malformed = [b"FUGG" + data[4:], data[:4] + struct.pack("<I", 2) + data[8:], data[:22]]
        for value in malformed:
            with self.subTest(value=value[:8]):
                with self.assertRaises(model_info.GGUFError):
                    self.inspect(value)

    def test_excessive_count_and_string_length_fail_before_allocation(self):
        cases = [
            b"GGUF" + struct.pack("<IQQ", 3, model_info.MAX_TENSORS + 1, 1),
            b"GGUF" + struct.pack("<IQQ", 3, 1, model_info.MAX_METADATA_ITEMS + 1),
            b"GGUF" + struct.pack("<IQQQ", 3, 1, 1, (1 << 63)),
        ]
        for data in cases:
            with self.subTest(length=len(data)):
                with self.assertRaises(model_info.GGUFError):
                    self.inspect(data)

    def test_array_limits_and_nested_arrays(self):
        for kind, count in ((9, 1), (8, model_info.MAX_ARRAY_ITEMS + 1)):
            metadata = text_bytes("test.array") + struct.pack("<IIQ", 9, kind, count)
            with self.subTest(kind=kind):
                with self.assertRaises(model_info.GGUFError):
                    self.inspect(tiny_gguf(metadata=[metadata]))

    def test_duplicate_metadata_and_tensor_names(self):
        metadata = [item("general.architecture", 8, "qwen35")] * 2
        tensors = [("same", [4], 0, 0), ("same", [4], 0, 32)]
        for data in (tiny_gguf(metadata=metadata), tiny_gguf(tensors=tensors)):
            with self.assertRaises(model_info.GGUFError):
                self.inspect(data)

    def test_invalid_scalars_and_alignment(self):
        cases = [item("test.bool", 7, 2), item("test.float", 6, float("nan")), item("general.alignment", 4, 3)]
        for metadata in cases:
            with self.assertRaises(model_info.GGUFError):
                self.inspect(tiny_gguf(metadata=[metadata]))

    def test_tensor_extent_shape_and_type_rejections(self):
        cases = [
            [("invalid-shape", [0], 0, 0)],
            [("invalid-row", [255], 12, 0)],
            [("unsupported-type", [4], 1, 0)],
            [("unaligned", [4], 0, 1)],
            [("past-end", [4], 0, 96)],
            [("first", [16], 0, 0), ("overlap", [4], 0, 32)],
        ]
        for tensors in cases:
            with self.subTest(tensors=tensors):
                with self.assertRaises(model_info.GGUFError):
                    self.inspect(tiny_gguf(tensors=tensors))

    def test_header_budget(self):
        reader = model_info.HeaderReader(io.BytesIO(bytes(64)), 64)
        with mock.patch.object(model_info, "MAX_HEADER_BYTES", 32):
            with self.assertRaises(model_info.GGUFError):
                reader.read(33)
        self.assertEqual(reader.offset, 0)

    @unittest.skipUnless(hasattr(os, "mkfifo") and hasattr(os, "O_NONBLOCK"), "POSIX named pipe check")
    def test_named_pipe_is_rejected_without_waiting_for_a_writer(self):
        os.mkfifo(self.path)
        with self.assertRaisesRegex(model_info.GGUFError, "regular file"):
            model_info.inspect_gguf(self.path)

    def test_checksum_streaming_and_changed_file_detection(self):
        data = b"synthetic checksum bytes" * 200_000
        self.path.write_bytes(data)
        self.assertEqual(model_info.hash_file(self.path), hashlib.sha256(data).hexdigest())
        old_stat = model_info._stat_signature(self.path.stat())
        self.path.write_bytes(b"changed")
        with self.assertRaisesRegex(model_info.GGUFError, "changed between"):
            model_info.hash_file(self.path, old_stat)

    def test_cli_reports_mismatch_and_never_hashes_rejected_structure(self):
        self.path.write_bytes(tiny_gguf())
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()), mock.patch.object(model_info, "hash_file") as checksum:
            status = model_info.main([str(self.path), "--json", "--verify-sha256"])
        result = json.loads(stdout.getvalue())
        self.assertEqual(status, 2)
        self.assertFalse(result["structural_match"])
        self.assertEqual(result["checkpoint_identity"], "unverified")
        checksum.assert_not_called()


class TargetGuardTests(unittest.TestCase):
    def setUp(self):
        # This is a synthetic observation to exercise classifier logic, not a GGUF
        # weight fixture or evidence that the real checkpoint can run.
        self.profile = copy.deepcopy(model_info.load_profile())
        synthetic_template = "synthetic template for classifier tests only"
        self.profile["gguf"]["chat_template_sha256"] = hashlib.sha256(synthetic_template.encode()).hexdigest()
        expected = self.profile["gguf"]
        self.info = {key: expected[key] for key in ("version", "metadata_count", "tensor_count", "metadata_end_offset", "tensor_table_end_offset", "tensor_data_offset", "tensor_descriptor_sha256")}
        self.info.update({
            "file_size": self.profile["artifact"]["size_bytes"],
            "metadata": copy.deepcopy(expected["metadata"]),
            "arrays": copy.deepcopy(expected["arrays"]),
            "special_tokens": copy.deepcopy(expected["special_tokens"]),
            "tensors": model_info.expected_tensors(self.profile),
            "stat_signature": (),
        })
        self.info["metadata"]["tokenizer.chat_template"] = synthetic_template

    def errors(self):
        return model_info.validate_target(self.info, self.profile)

    def test_declared_65_and_mtp_one_accept_exactly_64_recurrent_blocks(self):
        self.assertEqual(self.info["metadata"]["qwen35.block_count"], 65)
        self.assertEqual(self.info["metadata"]["qwen35.nextn_predict_layers"], 1)
        self.assertEqual(self.info["metadata"]["qwen35.full_attention_interval"], 4)
        self.assertEqual(len(self.info["tensors"]), 899)
        self.assertFalse(any(name.startswith("blk.64.") for name in self.info["tensors"]))
        self.assertEqual(self.errors(), [])

    def test_missing_middle_recurrent_layer_is_not_accepted_by_name(self):
        del self.info["tensors"]["blk.31.ssm_conv1d.weight"]
        self.assertTrue(any("blk.31.ssm_conv1d.weight" in error for error in self.errors()))

    def test_wrong_geometry_quantization_or_unexpected_mtp_is_rejected(self):
        for change in ("shape", "type", "mtp"):
            with self.subTest(change=change):
                original = copy.deepcopy(self.info)
                if change == "shape":
                    self.info["tensors"]["blk.63.ssm_norm.weight"]["dimensions"] = [64]
                elif change == "type":
                    self.info["tensors"]["blk.0.attn_qkv.weight"]["ggml_type"] = 12
                else:
                    self.info["tensors"]["blk.64.attn_norm.weight"] = {"dimensions": [5120], "ggml_type": 0}
                self.assertTrue(self.errors())
                self.info = original

    def test_changed_template_or_special_tokens_are_rejected(self):
        self.info["metadata"]["tokenizer.chat_template"] += " changed"
        self.info["special_tokens"]["248046"] = "<wrong>"
        errors = self.errors()
        self.assertTrue(any("chat template" in error for error in errors))
        self.assertTrue(any("Special token" in error for error in errors))

    def test_structural_match_never_implies_weight_identity(self):
        stdout = io.StringIO()
        with mock.patch.object(model_info, "load_profile", return_value=self.profile), mock.patch.object(model_info, "inspect_gguf", return_value=self.info), mock.patch.object(model_info, "hash_file") as checksum, contextlib.redirect_stdout(stdout):
            status = model_info.main(["synthetic-no-weights.gguf", "--json"])
        result = json.loads(stdout.getvalue())
        self.assertEqual(status, 0)
        self.assertTrue(result["structural_match"])
        self.assertEqual(result["checkpoint_identity"], "unverified")
        checksum.assert_not_called()

    def test_checksum_mismatch_fails_even_when_structure_matches(self):
        stdout = io.StringIO()
        with mock.patch.object(model_info, "load_profile", return_value=self.profile), mock.patch.object(model_info, "inspect_gguf", return_value=self.info), mock.patch.object(model_info, "hash_file", return_value="0" * 64), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            status = model_info.main(["synthetic-no-weights.gguf", "--json", "--verify-sha256"])
        result = json.loads(stdout.getvalue())
        self.assertEqual(status, 2)
        self.assertTrue(result["structural_match"])
        self.assertEqual(result["checkpoint_identity"], "mismatch")


if __name__ == "__main__":
    unittest.main()
