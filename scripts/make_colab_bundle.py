"""Build colab_bundle.zip with only what scripts/train_bert_colab.py needs, and print the
commands to run it on Colab.

    python scripts/make_colab_bundle.py

Paths inside the zip mirror the repo, so the training script's defaults
(--data data/training_data.json) work unchanged after unzipping.
"""
import argparse
import ast
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ENTRY = Path("scripts/train_bert_colab.py")
EXTRA_FILES = [Path("requirements-train.txt"), Path("data/training_data.json")]

COLAB_COMMANDS = """\
# 1. Runtime > Change runtime type > T4 GPU (or better).
# 2. Upload {zip_name} with the Files sidebar (or copy it from Drive), then in a cell:
!unzip -q -o /content/{zip_name} -d /content/sous
%cd /content/sous
!pip install -q -r requirements-train.txt

# 3. The two runs (each writes its own folder and metrics file):
!python scripts/train_bert_colab.py                 # runs/original/model, Results/metrics_original.json
!python scripts/train_bert_colab.py --fix-labels    # runs/fixed/model,    Results/metrics_fixed.json

# 4. Download the metrics and both zipped models:
!zip -q -r colab_outputs.zip Results runs/original/model.zip runs/fixed/model.zip
from google.colab import files
files.download("colab_outputs.zip")
"""


def local_imports(path, search_dir):
    """Modules imported by `path` that are files in `search_dir`, followed recursively."""
    found, todo = set(), [path]
    while todo:
        tree = ast.parse((REPO / todo.pop()).read_text())
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names.add(node.module.split(".")[0])
        for name in names:
            candidate = search_dir / f"{name}.py"
            if (REPO / candidate).exists() and candidate not in found and candidate != path:
                found.add(candidate)
                todo.append(candidate)
    return sorted(found)


def bundle_files():
    return [ENTRY, *local_imports(ENTRY, ENTRY.parent), *EXTRA_FILES]


def build_bundle(out_path):
    files = bundle_files()
    missing = [str(f) for f in files if not (REPO / f).exists()]
    if missing:
        raise SystemExit(f"Missing files: {', '.join(missing)}")
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(REPO / f, arcname=f.as_posix())
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(REPO / "colab_bundle.zip"))
    args = parser.parse_args()

    out = Path(args.out)
    files = build_bundle(out)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB):")
    for f in files:
        print(f"  {f.as_posix():<35} {(REPO / f).stat().st_size / 1e6:8.2f} MB")
    print("\nOn Colab:\n")
    print(COLAB_COMMANDS.format(zip_name=out.name))


if __name__ == "__main__":
    main()
