import zipfile
from pathlib import Path

import pytest

from make_colab_bundle import REPO, build_bundle, bundle_files, local_imports


def test_bundle_contains_only_training_files():
    assert [f.as_posix() for f in bundle_files()] == [
        "scripts/train_bert_colab.py", "requirements-train.txt", "data/training_data.json",
    ]


def test_local_imports_follow_repo_modules():
    # evaluate_gold imports gold_set, which imports train_bert_colab
    assert [p.name for p in local_imports(Path("scripts/evaluate_gold.py"), Path("scripts"))] == [
        "gold_set.py", "train_bert_colab.py",
    ]


@pytest.mark.skipif(not (REPO / "data/training_data.json").exists(), reason="training data not present")
def test_build_bundle_writes_repo_relative_paths(tmp_path):
    out = tmp_path / "bundle.zip"
    build_bundle(out)
    with zipfile.ZipFile(out) as zf:
        assert sorted(zf.namelist()) == [
            "data/training_data.json", "requirements-train.txt", "scripts/train_bert_colab.py",
        ]
        assert zf.read("scripts/train_bert_colab.py") == (REPO / "scripts/train_bert_colab.py").read_bytes()
