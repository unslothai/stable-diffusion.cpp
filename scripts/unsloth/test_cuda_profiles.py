#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright 2026-present the Unsloth AI Inc. team.

"""package_bundle.py + assemble_metadata.py for the per-GPU-family CUDA bundles, and the workflow's
profile table. Run: python3 scripts/unsloth/test_cuda_profiles.py"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import package_bundle  # noqa: E402

WORKFLOW = HERE.parents[1] / ".github" / "workflows" / "unsloth-sd-prebuilt.yml"


def _package(tmp: Path, env: dict) -> Path:
    bin_dir = tmp / "bin"
    bin_dir.mkdir(exist_ok = True)
    (bin_dir / "sd-cli").write_bytes(b"cli")
    (bin_dir / "sd-server").write_bytes(b"server")
    (bin_dir / "libggml-cuda.so").write_bytes(b"cuda")
    full = {"BIN_DIR": str(bin_dir), "OUT_DIR": str(tmp / "dist"), "TAG": "master-1-abc-udef",
            "LABEL": "x", "PATH": os.environ.get("PATH", "")}
    full.update(env)
    subprocess.run([sys.executable, str(HERE / "package_bundle.py")], env = full, check = True,
                   stdout = subprocess.DEVNULL)
    return tmp / "dist"


def _profile_env(asset: str, fmt: str, profile: str, klass: str, rank: int, archs: str, os_: str) -> dict:
    return {"ASSET_STEM": f"sd-master-1-abc-udef-{asset}", "ARCHIVE_FORMAT": fmt, "PROFILE": profile,
            "RUNTIME_LINE": profile.split("-")[0], "COVERAGE_CLASS": klass, "RANK": str(rank),
            "ARCHS": archs, "PLATFORM_OS": os_, "PLATFORM_ARCH": "x64", "TOOLKIT": "12.8.0"}


class ProfileInfo(unittest.TestCase):
    def test_sass_and_ptx(self):
        info = package_bundle.cuda_profile_info("86-real;89-real;90-real;100;120-real")
        self.assertEqual(info["sass_sms"], ["86", "89", "90", "100", "120"])
        self.assertEqual(info["ptx_sms"], ["100"])
        self.assertEqual(info["supported_sms"], ["86", "89", "90", "100", "103", "120"])
        self.assertEqual((info["min_sm"], info["max_sm"]), (86, 120))

    def test_bare_arch_is_sass_and_ptx_but_12x_ptx_is_arch_specific(self):
        info = package_bundle.cuda_profile_info("75;120")
        self.assertEqual(info["sass_sms"], ["75", "120"])
        self.assertEqual(info["ptx_sms"], ["75"])

    def test_virtual_only_needs_sms(self):
        with self.assertRaises(SystemExit):
            package_bundle.cuda_profile_info("50-virtual;61-virtual")
        info = package_bundle.cuda_profile_info("50-virtual;61-virtual", "50 52 60 61")
        self.assertEqual(info["supported_sms"], ["50", "52", "60", "61"])
        self.assertEqual(info["ptx_sms"], ["50", "61"])


class Bundles(unittest.TestCase):
    def test_split_tar_and_portable_zip_reach_the_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            _package(tmp, _profile_env("linux-x64-cuda12-older", "tar.gz", "cuda12-older", "older", 10,
                                       "70-real;75-real;80-real;86-real;89", "linux"))
            _package(tmp, _profile_env("bin-Linux-Ubuntu-22.04-x86_64-cuda12", "zip", "cuda12-portable",
                                       "portable", 30, "70-real;75-real;80-real;86-real;89-real;90-real;100;120-real",
                                       "linux"))
            dist = _package(tmp, {"LABEL": "win-cpu-x64"})
            tar_path = dist / "sd-master-1-abc-udef-linux-x64-cuda12-older.tar.gz"
            with tarfile.open(tar_path) as tf:
                members = tf.getmembers()
            self.assertTrue(all(m.isfile() for m in members))
            self.assertIn("sd-master-1-abc-udef-linux-x64-cuda12-older/UNSLOTH_PREBUILT_INFO.json",
                          [m.name for m in members])
            with zipfile.ZipFile(dist / "sd-master-1-abc-udef-bin-win-cpu-x64.zip") as zf:
                self.assertFalse(any(n.endswith("UNSLOTH_PREBUILT_INFO.json") for n in zf.namelist()))
            subprocess.run([sys.executable, str(HERE / "assemble_metadata.py"), "--tag", "master-1-abc-udef",
                            "--source-repo", "o/r", "--commit", "c" * 40, "--dist", str(dist), "--out", str(dist),
                            "--publish-repo", "o/r"], check = True, stdout = subprocess.DEVNULL)
            manifest = json.loads((dist / "sd-prebuilt-manifest.json").read_text())
            sha = json.loads((dist / "sd-prebuilt-sha256.json").read_text())
            by_name = {a["name"]: a for a in manifest["assets"]}
            self.assertEqual(set(by_name), set(sha))
            older = by_name["sd-master-1-abc-udef-linux-x64-cuda12-older.tar.gz"]
            self.assertEqual((older["coverage_class"], older["rank"], older["runtime_line"]), ("older", 10, "cuda12"))
            self.assertEqual((older["os"], older["arch"], older["label"]), ("linux", "x64", "linux-x64-cuda12-older"))
            self.assertEqual(older["ptx_sms"], ["89"])
            portable = by_name["sd-master-1-abc-udef-bin-Linux-Ubuntu-22.04-x86_64-cuda12.zip"]
            self.assertEqual(portable["coverage_class"], "portable")
            self.assertIn("103", portable["supported_sms"])
            self.assertNotIn("supported_sms", by_name["sd-master-1-abc-udef-bin-win-cpu-x64.zip"])


class WorkflowTable(unittest.TestCase):
    def test_profiles_match_between_linux_and_windows(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not installed")
        wf = yaml.safe_load(WORKFLOW.read_text())
        rows = {}
        for job in ("build-linux-cuda", "build-windows-cuda"):
            rows[job] = {r["profile"]: (r["line"], r["cuda"], r["klass"], r["rank"], r["archs"])
                         for r in wf["jobs"][job]["strategy"]["matrix"]["include"]}
        self.assertEqual(rows["build-linux-cuda"], rows["build-windows-cuda"])
        for profile, (line, _cuda, klass, _rank, archs) in rows["build-linux-cuda"].items():
            info = package_bundle.cuda_profile_info(archs)
            # SASS for every arch, PTX for exactly one: the newest generic one.
            self.assertEqual(len(info["ptx_sms"]), 1, profile)
            self.assertEqual(profile, f"{line}-{klass}")
        portable = {p: package_bundle.cuda_profile_info(r[4]) for p, r in rows["build-linux-cuda"].items()
                    if r[2] == "portable"}
        for profile, (line, *_rest, archs) in rows["build-linux-cuda"].items():
            sass = set(package_bundle.cuda_profile_info(archs)["sass_sms"])
            self.assertTrue(sass <= set(portable[f"{line}-portable"]["sass_sms"]), profile)


if __name__ == "__main__":
    unittest.main()
