#!/usr/bin/env python3
"""Run the actual CPU engine with random synthetic GGUF data, never real weights.

Usage: python3 tests/test_native.py /absolute/path/to/longbook27
Or set LONGBOOK27_BIN when running Python unittest discovery. Without a binary
these integration tests are explicitly skipped; a supplied invalid path fails.

Passing these checks is evidence for loader/graph/budget behavior only. It does
not validate the downloaded Longbook27 checkpoint, quality, VL, or GPU backends.
"""

from __future__ import annotations

import json
import math
import os
import re
import selectors
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from make_tiny_gguf import (
    CONTEXT_LENGTH,
    DECLARED_MTP_BLOCKS,
    EMBEDDING_LENGTH,
    RECURRENT_BLOCKS,
    write_model,
)


NATIVE_BINARY = os.environ.get("LONGBOOK27_BIN")
STATS = re.compile(
    r"prompt_tokens=(\d+)\s+generated_tokens=(\d+)\s+stop=(\w+)\s+"
    r"prefill_ms=(\S+)\s+generation_ms=(\S+)"
)


class NativeSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not NATIVE_BINARY:
            raise unittest.SkipTest("set LONGBOOK27_BIN or pass a built native executable")
        cls.binary = Path(NATIVE_BINARY).resolve(strict=True)
        if not cls.binary.is_file() or not os.access(cls.binary, os.X_OK):
            raise RuntimeError(f"native executable is not executable: {cls.binary}")
        cls.temp = tempfile.TemporaryDirectory(prefix="longbook27-native-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.model = write_model(Path(cls.temp.name) / "synthetic-qwen35.gguf")

    def invoke(self, *args: str, compatible: bool = True, stdin: str | None = None) -> subprocess.CompletedProcess:
        command = [str(self.binary), "--model", str(self.model)]
        if compatible:
            command.append("--allow-compatible-model")
        command.extend(args)
        return subprocess.run(
            command, input=stdin, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=60, check=False,
        )

    def generate(self, prompt: str, *, context: int = 128, predict: int = 8,
                 batch: int = 32, ubatch: int = 8) -> subprocess.CompletedProcess:
        return self.invoke(
            "--raw", "--prompt", prompt,
            "--ctx-size", str(context), "--predict", str(predict),
            "--batch-size", str(batch), "--ubatch-size", str(ubatch),
            "--gpu-layers", "0", "--threads", "1", "--seed", "27", "--temp", "0",
        )

    def assert_generation(self, result: subprocess.CompletedProcess, prompt: str, predicted: int) -> str:
        self.assertEqual(result.returncode, 0, result.stderr)
        match = STATS.search(result.stderr)
        self.assertIsNotNone(match, f"missing generation accounting:\n{result.stderr}")
        prompt_tokens, generated_tokens, stop, prefill_ms, generation_ms = match.groups()
        # With the byte alphabet, empty merge list and disabled BOS/EOS, every
        # UTF-8 byte is one token. This independently detects dropped input.
        self.assertEqual(int(prompt_tokens), len(prompt.encode("utf-8")))
        self.assertEqual(int(generated_tokens), predicted)
        self.assertEqual(stop, "limit")
        for duration in (prefill_ms, generation_ms):
            self.assertTrue(math.isfinite(float(duration)))
            self.assertGreaterEqual(float(duration), 0.0)
        output = result.stdout.rstrip("\r\n")
        self.assertEqual(len(output), predicted, result.stdout)
        self.assertTrue(all(33 <= ord(char) <= 126 for char in output), repr(output))
        return output

    def test_inspection_infers_all_recurrent_despite_stale_interval(self) -> None:
        result = self.invoke("--inspect")
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads(result.stdout)
        self.assertEqual(metadata["architecture"], "qwen35")
        self.assertEqual(metadata["gguf_version"], 3)
        self.assertEqual(metadata["tensor_count"], 3 + RECURRENT_BLOCKS * 14)
        self.assertEqual(metadata["declared_blocks"], RECURRENT_BLOCKS + DECLARED_MTP_BLOCKS)
        self.assertEqual(metadata["declared_mtp_blocks"], DECLARED_MTP_BLOCKS)
        self.assertEqual(metadata["recurrent_blocks"], RECURRENT_BLOCKS)
        self.assertEqual(metadata["context_length"], CONTEXT_LENGTH)
        self.assertEqual(metadata["embedding_length"], EMBEDDING_LENGTH)
        self.assertIs(metadata["longbook27_signature"], False)
        self.assertIs(metadata["weights_verified"], False)
        self.assertRegex(metadata["engine_commit"], r"^[0-9a-f]{40}$")

    def test_default_guard_rejects_the_synthetic_model(self) -> None:
        result = self.invoke(
            "--raw", "--prompt", "x", "--ctx-size", "128", "--predict", "1",
            compatible=False,
        )
        self.assertGreater(result.returncode, 0, result.stderr)
        self.assertIn("Longbook27 signature mismatch", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_recurrent_decode_agrees_across_batch_boundaries(self) -> None:
        # 113 prompt tokens exceed even a backend's usual 64-token minimum batch;
        # the smaller ubatch also exercises partial recurrent-state updates.
        prompt = ("alpha beta delta gamma. " * 6)[:113]
        split = self.generate(prompt, batch=32, ubatch=8)
        one_batch = self.generate(prompt, batch=128, ubatch=128)
        repeated = self.generate(prompt, batch=32, ubatch=8)
        output = self.assert_generation(split, prompt, 8)
        self.assertEqual(self.assert_generation(one_batch, prompt, 8), output)
        self.assertEqual(self.assert_generation(repeated, prompt, 8), output)

    def test_exact_context_budget_succeeds(self) -> None:
        prompt = "a" * 28
        result = self.generate(prompt, context=32, predict=4, batch=32, ubatch=8)
        self.assert_generation(result, prompt, 4)

    def test_context_budget_overflow_is_rejected_without_truncation(self) -> None:
        result = self.generate("a" * 29, context=32, predict=4, batch=32, ubatch=8)
        self.assertGreater(result.returncode, 0, result.stderr)
        self.assertIn("context budget exceeded", result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertIsNone(STATS.search(result.stderr), "overflow must stop before generation")

    def test_metadata_context_limit_is_enforced(self) -> None:
        result = self.generate("x", context=CONTEXT_LENGTH + 1, predict=1)
        self.assertGreater(result.returncode, 0, result.stderr)
        self.assertIn("context", result.stderr.lower())
        self.assertEqual(result.stdout, "")
        self.assertIsNone(STATS.search(result.stderr))

    def test_utf8_stdin_preserves_line_breaks_and_bytes(self) -> None:
        prompt = "日本語と UTF-8: 🐉\n\nSecond line.\n"
        result = self.invoke(
            "--raw", "--prompt-file", "-", "--ctx-size", "128", "--predict", "4",
            "--batch-size", "32", "--ubatch-size", "8", "--gpu-layers", "0",
            "--threads", "1", "--seed", "27", "--temp", "0", stdin=prompt,
        )
        self.assert_generation(result, prompt, 4)

    def test_interactive_reset_reproduces_first_turn_and_eof_exits(self) -> None:
        result = self.invoke(
            "--interactive", "--ctx-size", "256", "--predict", "4",
            "--batch-size", "32", "--ubatch-size", "8", "--gpu-layers", "0",
            "--threads", "1", "--seed", "27", "--temp", "0",
            stdin="repeatable\n/reset\nrepeatable\n",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        turns = STATS.findall(result.stderr)
        self.assertEqual(len(turns), 2, result.stderr)
        self.assertEqual(turns[0][:3], turns[1][:3])
        self.assertEqual(turns[0][1:3], ("4", "limit"))
        output_lines = result.stdout.splitlines()
        self.assertEqual(len(output_lines), 2, repr(result.stdout))
        self.assertEqual(len(output_lines[0]), 4)
        self.assertEqual(output_lines[0], output_lines[1])

    @unittest.skipUnless(os.name == "posix", "requires POSIX signal and pipe behavior")
    def test_sigterm_exits_while_interactive_stdin_remains_open(self) -> None:
        command = [
            str(self.binary), "--model", str(self.model), "--allow-compatible-model",
            "--interactive", "--ctx-size", "256", "--predict", "4",
            "--gpu-layers", "0", "--threads", "1", "--temp", "0",
        ]
        with subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0,
        ) as process:
            diagnostics = bytearray()
            try:
                deadline = time.monotonic() + 5.0
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stderr, selectors.EVENT_READ)
                    while b"you> " not in diagnostics:
                        remaining = deadline - time.monotonic()
                        self.assertGreater(remaining, 0, "interactive input prompt was not reached")
                        events = selector.select(timeout=remaining)
                        self.assertTrue(events, "timed out waiting for interactive input prompt")
                        chunk = os.read(process.stderr.fileno(), 4096)
                        self.assertTrue(chunk, diagnostics.decode("utf-8", errors="replace"))
                        diagnostics.extend(chunk)

                process.send_signal(signal.SIGTERM)
                # communicate() closes stdin. Wait for exit first so EOF cannot
                # hide a regression that leaves getline() blocked after SIGTERM.
                process.wait(timeout=3)
                stdout, stderr = process.communicate(timeout=3)
                diagnostics.extend(stderr)
                diagnostic_text = diagnostics.decode("utf-8", errors="replace")
                self.assertEqual(process.returncode, 128 + signal.SIGTERM, diagnostic_text)
                self.assertEqual(stdout, b"")
                self.assertIsNone(STATS.search(diagnostic_text), diagnostic_text)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=3)


if __name__ == "__main__":
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        NATIVE_BINARY = sys.argv.pop(1)
    unittest.main(verbosity=2)
