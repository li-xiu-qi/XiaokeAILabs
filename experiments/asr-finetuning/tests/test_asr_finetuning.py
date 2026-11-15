# -*- coding: utf-8 -*-
"""离线单元测试：不加载模型、不下载数据、不需要 GPU。

只覆盖 asr_common 的纯 Python 逻辑（归一化、manifest、集合隔离、输出目录
保护、环境变量契约）。运行：python -m unittest discover -s tests -v
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import asr_common as common  # noqa: E402


def write_jsonl(path, rows):
    Path(path).write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )


class NormalizeTextTest(unittest.TestCase):
    def test_case_and_punctuation(self):
        self.assertEqual(common.normalize_text("Hello, WORLD!"), "hello world")

    def test_apostrophe_and_digits_kept(self):
        self.assertEqual(common.normalize_text("It's 1990."), "it's 1990")

    def test_whitespace_collapsed(self):
        self.assertEqual(common.normalize_text("  multiple   spaces "), "multiple spaces")

    def test_curly_quotes(self):
        self.assertEqual(common.normalize_text("‘Hi’"), "'hi'")

    def test_non_ascii_dropped(self):
        self.assertEqual(common.normalize_text("café"), "caf")


class ManifestTest(unittest.TestCase):
    def test_read_manifest_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.jsonl"
            write_jsonl(path, [{"id": "a", "text": "hello"}, {"id": "b", "text": "world"}])
            rows = common.read_manifest(path)
            self.assertEqual([r["id"] for r in rows], ["a", "b"])

    def test_read_manifest_rejects_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            path.write_text('{"id": "a"}\nnot-json\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                common.read_manifest(path)

    def test_read_manifest_rejects_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.jsonl"
            path.write_text("\n\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                common.read_manifest(path)

    def test_split_isolation_detects_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            train = Path(tmp) / "train.jsonl"
            test = Path(tmp) / "test.jsonl"
            write_jsonl(train, [{"id": "a"}, {"id": "b"}])
            write_jsonl(test, [{"id": "b"}, {"id": "c"}])
            with self.assertRaises(ValueError):
                common.check_split_isolation(train, test)

    def test_split_isolation_accepts_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            train = Path(tmp) / "train.jsonl"
            test = Path(tmp) / "test.jsonl"
            write_jsonl(train, [{"id": "a"}, {"id": "b"}])
            write_jsonl(test, [{"id": "c"}])
            common.check_split_isolation(train, test)

    def test_filter_by_duration(self):
        rows = [{"id": "a", "duration": 0.2}, {"id": "b", "duration": 5.0},
                {"id": "c", "duration": 31.0}, {"id": "d", "duration": "oops"}]
        kept, dropped = common.filter_by_duration(rows, 0.5, 30.0)
        self.assertEqual([r["id"] for r in kept], ["b"])
        self.assertEqual(dropped, 3)


class OutputProtectionTest(unittest.TestCase):
    def test_fresh_dir_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "nested" / "out"
            common.ensure_fresh_dir(target)
            self.assertTrue(target.is_dir())

    def test_non_empty_dir_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "out"
            target.mkdir()
            (target / "keep.txt").write_text("x", encoding="utf-8")
            with self.assertRaises(ValueError):
                common.ensure_fresh_dir(target)

    def test_regular_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "out"
            target.write_text("x", encoding="utf-8")
            with self.assertRaises(ValueError):
                common.ensure_fresh_dir(target)

    def test_inputs_inside_output_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            out.mkdir()
            inner = out / "train.jsonl"
            with self.assertRaises(ValueError):
                common.ensure_paths_outside(out, [inner])


class EnvContractTest(unittest.TestCase):
    def test_require_external_env_blocks_fallback(self):
        with mock.patch.dict(os.environ, {"XL_REQUIRE_EXTERNAL_ENV": "1"}, clear=False):
            os.environ.pop("XL_RUN_ID", None)
            with self.assertRaises(SystemExit):
                common.run_id()

    def test_require_external_env_allows_explicit(self):
        with mock.patch.dict(
            os.environ, {"XL_REQUIRE_EXTERNAL_ENV": "1", "XL_RUN_ID": "run-001"}, clear=False
        ):
            self.assertEqual(common.run_id(), "run-001")

    def test_local_fallback_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("XL_RUN_ID", None)
            os.environ.pop("XL_REQUIRE_EXTERNAL_ENV", None)
            self.assertEqual(common.run_id(), "local")

    def test_model_ref_requires_revision_by_default(self):
        with mock.patch.dict(os.environ, {"ASR_MODEL": "openai/whisper-tiny"}, clear=False):
            os.environ.pop("ASR_MODEL_REVISION", None)
            os.environ.pop("ASR_REQUIRE_REVISION", None)
            with self.assertRaises(SystemExit):
                common.model_ref()

    def test_model_ref_returns_pair(self):
        with mock.patch.dict(
            os.environ,
            {"ASR_MODEL": "openai/whisper-tiny", "ASR_MODEL_REVISION": "abc123"},
            clear=False,
        ):
            self.assertEqual(common.model_ref(), ("openai/whisper-tiny", "abc123"))

    def test_model_ref_floating_opt_out(self):
        with mock.patch.dict(
            os.environ,
            {"ASR_MODEL": "openai/whisper-tiny", "ASR_REQUIRE_REVISION": "0"},
            clear=False,
        ):
            os.environ.pop("ASR_MODEL_REVISION", None)
            self.assertEqual(common.model_ref(), ("openai/whisper-tiny", None))


class WerTest(unittest.TestCase):
    def test_wer_percent_basic(self):
        try:
            import jiwer  # noqa: F401
        except ImportError:
            self.skipTest("jiwer 未安装")
        value = common.wer_percent(["a b c"], ["a b d"])
        self.assertAlmostEqual(value, 100.0 / 3.0, places=4)

    def test_wer_percent_rejects_empty_reference(self):
        try:
            import jiwer  # noqa: F401
        except ImportError:
            self.skipTest("jiwer 未安装")
        with self.assertRaises(ValueError):
            common.wer_percent([""], [""])


if __name__ == "__main__":
    unittest.main()
