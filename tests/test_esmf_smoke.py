"""
Smoke test for ESMFold backend: CLI and output layout.

Runs in CPU mode on a tiny sequence so CI stays fast.
Skips actual ESMFold inference if fair-esm is not installed.

Run with: pytest tests/test_esmf_smoke.py -v
Or without pytest: python tests/test_esmf_smoke.py
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pytest
except ImportError:
    pytest = None


def _has_esm() -> bool:
    try:
        import esm  # noqa: F401
        return True
    except ImportError:
        return False


def test_esmf_import():
    """Backend and runner can be imported when fair-esm is present."""
    if not _has_esm():
        return  # skip when no fair-esm
    from vizfold.backends.esmfold.inference import ESMFoldRunner
    from vizfold.backends.esmfold.schema import build_meta, write_meta
    assert ESMFoldRunner is not None
    meta = build_meta(
        backend="esmfold",
        model_name="esmfold_v1",
        out_dir="/tmp",
        fasta_path=None,
        device="cpu",
        dtype="float32",
        sequence_length=10,
        fasta_hash="abc",
        layer_count=1,
        head_count=4,
        trace_mode="none",
        trace_formats=[],
        shapes_recorded={},
    )
    assert meta["backend"] == "esmfold"
    assert "date_time" in meta


def test_cli_help():
    """CLI runs and shows help."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "run_pretrained_esmf.py"), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--fasta" in result.stdout
    assert "--trace_mode" in result.stdout


def test_cli_missing_fasta():
    """CLI exits non-zero when FASTA is missing."""
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "run_pretrained_esmf.py"),
            "--fasta", "/nonexistent.fasta",
            "--out", "/tmp/out_esmf_smoke",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "not found" in result.stderr or "Error" in result.stderr


def test_esmf_smoke_run_cpu(tmp_path=None):
    """
    Run ESMFold on a tiny sequence (CPU), check output layout.
    Kept minimal so it stays under a few minutes.
    """
    if not _has_esm():
        return  # skip when no fair-esm
    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp())
    fasta = tmp_path / "tiny.fasta"
    fasta.write_text(">tiny\nMKFLKFSLLTAVLLSVVFAFSSCGDDDD\n")
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "run_pretrained_esmf.py"),
            "--fasta", str(fasta),
            "--out", str(out_dir),
            "--device", "cpu",
            "--trace_mode", "none",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)

    assert (out_dir / "meta.json").exists()
    assert (out_dir / "logs.txt").exists()
    # Structure may or may not exist depending on model output
    structure_dir = out_dir / "structure"
    if structure_dir.exists():
        assert list(structure_dir.iterdir())


# Pytest decorators when available
if pytest is not None:
    test_esmf_import = pytest.mark.skipif(not _has_esm(), reason="fair-esm not installed")(test_esmf_import)
    test_esmf_smoke_run_cpu = pytest.mark.skipif(not _has_esm(), reason="fair-esm not installed")(test_esmf_smoke_run_cpu)


if __name__ == "__main__":
    # Run without pytest
    failed = []
    # CLI help
    print("test_cli_help ...", end=" ")
    try:
        test_cli_help()
        print("ok")
    except AssertionError as e:
        print("FAIL", e)
        failed.append("test_cli_help")
    # CLI missing fasta
    print("test_cli_missing_fasta ...", end=" ")
    try:
        test_cli_missing_fasta()
        print("ok")
    except AssertionError as e:
        print("FAIL", e)
        failed.append("test_cli_missing_fasta")
    # Schema/build_meta (no ESM needed)
    print("test_schema_build_meta ...", end=" ")
    try:
        from vizfold.backends.esmfold.schema import build_meta
        meta = build_meta(
            backend="esmfold", model_name="x", out_dir="/tmp", fasta_path=None,
            device="cpu", dtype="float32", sequence_length=10, fasta_hash="h",
            layer_count=1, head_count=4, trace_mode="none", trace_formats=[],
            shapes_recorded={},
        )
        assert meta["backend"] == "esmfold" and "date_time" in meta
        print("ok")
    except Exception as e:
        print("FAIL", e)
        failed.append("test_schema_build_meta")
    # ESMFold import (when fair-esm present)
    print("test_esmf_import ...", end=" ")
    try:
        test_esmf_import()
        print("ok (or skipped)")
    except Exception as e:
        print("FAIL", e)
        failed.append("test_esmf_import")
    # Full smoke run (when fair-esm present)
    print("test_esmf_smoke_run_cpu ...", end=" ")
    try:
        test_esmf_smoke_run_cpu()
        print("ok (or skipped)")
    except Exception as e:
        print("FAIL", e)
        failed.append("test_esmf_smoke_run_cpu")
    if failed:
        print("Failed:", failed)
        sys.exit(1)
    print("All checks passed.")
