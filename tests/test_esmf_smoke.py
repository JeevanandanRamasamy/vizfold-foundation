"""
Smoke test for ESMFold backend: CLI, schema, and output layout.

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
    assert ESMFoldRunner is not None


def test_schema_build_meta():
    """schema.build_meta returns valid dict with required fields."""
    from vizfold.backends.esmfold.schema import build_meta
    meta = build_meta(
        backend="esmfold",
        model_name="esmfold_v1",
        out_dir="/tmp",
        fasta_path=None,
        device="cpu",
        dtype="float32",
        sequence_length=10,
        fasta_hash="abc123",
        layer_count=48,
        head_count=8,
        trace_mode="attention",
        trace_formats=["txt"],
        shapes_recorded={},
    )
    assert meta["backend"] == "esmfold"
    assert "date_time" in meta
    assert meta["layer_count"] == 48
    assert meta["head_count"] == 8
    assert "msa_row_attn" in meta["attention_types"]
    assert "notes" in meta  # should document no triangle attention


def test_schema_triangle_warning():
    """schema.build_meta warns about triangle_residue_idx for ESMFold."""
    from vizfold.backends.esmfold.schema import build_meta
    meta = build_meta(
        backend="esmfold",
        model_name="esmfold_v1",
        out_dir="/tmp",
        fasta_path=None,
        device="cpu",
        dtype="float32",
        sequence_length=10,
        fasta_hash="abc",
        layer_count=48,
        head_count=8,
        trace_mode="attention",
        trace_formats=["txt"],
        shapes_recorded={},
        triangle_residue_idx=18,
    )
    assert "triangle_attention_warning" in meta
    assert "ignored" in meta["triangle_attention_warning"].lower()


def test_schema_fasta_hash():
    """_read_fasta_and_hash parses FASTA and returns hash."""
    from vizfold.backends.esmfold.schema import _read_fasta_and_hash
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        f.write(">test\nMKFLKFSLLT\n")
        f.flush()
        seq, h = _read_fasta_and_hash(f.name)
    os.unlink(f.name)
    assert seq == "MKFLKFSLLT"
    assert len(h) == 16  # sha256 prefix


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
    assert "--top_k" in result.stdout
    assert "triangle" in result.stdout.lower()  # should mention triangle limitation


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


def test_write_structure():
    """write_structure creates a PDB file."""
    from vizfold.backends.esmfold.trace_adapter import write_structure
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_str = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        paths = write_structure(tmpdir, pdb_str)
        assert "pdb" in paths
        assert os.path.isfile(paths["pdb"])
        with open(paths["pdb"]) as f:
            assert "ATOM" in f.read()


def test_write_meta():
    """build_and_write_meta creates meta.json."""
    import json
    from vizfold.backends.esmfold.trace_adapter import build_and_write_meta
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a dummy FASTA for the fasta_path
        fasta_path = os.path.join(tmpdir, "test.fasta")
        with open(fasta_path, "w") as f:
            f.write(">test\nMKFLKFSLLT\n")
        path = build_and_write_meta(
            out_dir=tmpdir,
            model_name="esmfold_v1",
            fasta_path=fasta_path,
            device="cpu",
            dtype="float32",
            seq_len=10,
            fasta_hash="abc123",
            layer_count=48,
            head_count=8,
            trace_mode="attention",
            shapes_recorded={},
        )
        assert os.path.isfile(path)
        with open(path) as f:
            meta = json.load(f)
        assert meta["backend"] == "esmfold"
        assert meta["sequence_length"] == 10


def test_patch_esmfold_state_dict():
    """_patch_esmfold_state_dict correctly renames IPA keys."""
    from vizfold.backends.esmfold.inference import _patch_esmfold_state_dict
    import torch
    state_dict = {
        "trunk.structure_module.ipa.linear_q_points.weight": torch.zeros(1),
        "trunk.structure_module.ipa.linear_q_points.bias": torch.zeros(1),
        "trunk.structure_module.ipa.linear_kv_points.weight": torch.zeros(1),
        "trunk.structure_module.ipa.linear_kv_points.bias": torch.zeros(1),
        "other.key": torch.zeros(1),
    }
    patched = _patch_esmfold_state_dict(state_dict)
    assert "trunk.structure_module.ipa.linear_q_points.linear.weight" in patched
    assert "trunk.structure_module.ipa.linear_q_points.linear.bias" in patched
    assert "trunk.structure_module.ipa.linear_kv_points.linear.weight" in patched
    assert "trunk.structure_module.ipa.linear_kv_points.linear.bias" in patched
    assert "trunk.structure_module.ipa.linear_q_points.weight" not in patched
    assert "other.key" in patched  # should be untouched


def test_parse_layers_arg():
    """_parse_layers_arg handles various input formats."""
    from vizfold.backends.esmfold.inference import _parse_layers_arg
    assert _parse_layers_arg(None) is None
    assert _parse_layers_arg("all") is None
    assert _parse_layers_arg("0,1,2") == [0, 1, 2]
    assert _parse_layers_arg("0:3") == [0, 1, 2]
    assert _parse_layers_arg("0,5,10:12") == [0, 5, 10, 11]


# Pytest decorators when available
if pytest is not None:
    test_esmf_import = pytest.mark.skipif(
        not _has_esm(), reason="fair-esm not installed"
    )(test_esmf_import)


if __name__ == "__main__":
    # Run without pytest
    failed = []
    tests = [
        ("test_schema_build_meta", test_schema_build_meta),
        ("test_schema_triangle_warning", test_schema_triangle_warning),
        ("test_schema_fasta_hash", test_schema_fasta_hash),
        ("test_cli_help", test_cli_help),
        ("test_cli_missing_fasta", test_cli_missing_fasta),
        ("test_write_structure", test_write_structure),
        ("test_write_meta", test_write_meta),
        ("test_patch_esmfold_state_dict", test_patch_esmfold_state_dict),
        ("test_parse_layers_arg", test_parse_layers_arg),
        ("test_esmf_import", test_esmf_import),
    ]
    for name, fn in tests:
        print(f"{name} ...", end=" ")
        try:
            fn()
            print("ok")
        except AssertionError as e:
            print(f"FAIL: {e}")
            failed.append(name)
        except Exception as e:
            print(f"ERROR: {e}")
            failed.append(name)
    if failed:
        print(f"\nFailed: {failed}")
        sys.exit(1)
    print("\nAll checks passed.")
