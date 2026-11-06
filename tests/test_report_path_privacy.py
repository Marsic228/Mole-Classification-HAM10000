"""Privacy regression tests for HAM10000 report producers (stdlib only)."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import tempfile
import unittest


class ReportPathPrivacyTests(unittest.TestCase):
    def test_original_environment_exporter_paths(self):
        m = importlib.import_module("evaluation.audit_cv_folds")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            root.mkdir()
            internal = root / "reports" / "repro_audit" / "sample.csv"
            self.assertEqual(m.public_repo_path(root, root), ".")
            self.assertEqual(m.public_repo_path(internal, root), "reports/repro_audit/sample.csv")
            self.assertEqual(m.public_repo_path(root.parent / "private" / "python.exe", root), "<EXTERNAL_PATH>")
            self.assertEqual(m.public_repo_path(root.parent / "project-copy" / "foo", root), "<EXTERNAL_PATH>")

    def test_scanner_decodes_json_paths(self):
        m = importlib.import_module("scripts.prepublish_check")
        slash = chr(92)
        unsafe_path = slash.join(("C:", "Users", "ExampleAccount", "repo"))
        unsafe = json.dumps({"project_root": unsafe_path}).encode()
        labels = [kind for _, kind in m.inspect_text(unsafe, "reports/repro_audit/test.json")]
        self.assertTrue(any("Windows user-home path" in s for s in labels))
        self.assertTrue(any("decoded JSON" in s for s in labels))
        safe = json.dumps({"project_root": ".", "csv_file": "reports/repro_audit/folds.csv"}).encode()
        self.assertFalse(m.inspect_text(safe, "reports/repro_audit/test.json"))
        invalid = b'{"project_root":'
        self.assertIn((0, "invalid JSON"), m.inspect_text(invalid, "reports/x.json"))

    def test_canonical_manifest_paths(self):
        m = importlib.import_module("core.canonical_cv")
        root = Path(m.__file__).resolve().parents[1]
        self.assertEqual(m.public_artifact_path(root / "reports" / "test.csv"), "reports/test.csv")
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(m.public_artifact_path(Path(directory) / "local-secret.txt"), "<EXTERNAL_PATH>")


if __name__ == "__main__":
    unittest.main()
