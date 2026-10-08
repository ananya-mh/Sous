"""Hand-labeled gold set for the ingredient NER model.

Gold files are plain text, one "token<TAB>label" per line, blank line between sentences:

    # id: nyt-0001
    # text: 1 1/2 cups flour
    # reviewed: no
    1       B-AMT
    1/2     I-AMT
    cups    B-UNIT
    flour   B-NAME

Sample the NYT subset (only from the held-out test split of train_bert_colab.py, so no gold
sentence was seen in train or val), pre-labeled with fix_labels():

    python scripts/gold_set.py nyt --n 50
"""
import argparse
import random
import sys
from pathlib import Path

from train_bert_colab import LABEL_LIST, dedupe, fix_labels, load_sentences, split

HEADER = """\
# Sous NER gold set ({subset}). Pre-labeled by rules: correct every label by hand.
# One "token<TAB>label" per line (tabs or spaces), blank line between sentences.
# Labels: {labels}
# Conventions: one span per mixed number or range ("1 1/2", "2 to 3"); NAME is the item you
#   buy ("olive oil"); DESC is prep/state/size ("finely chopped"); punctuation and filler are O.
# Change "# reviewed: no" to "# reviewed: yes" once a sentence is checked.
# You may split or merge token lines if the tokenization itself is wrong.
"""


def write_gold(path, sentences, subset):
    """sentences: [{"id", "tokens", "tags"}]. Never overwrites: corrections are hand work."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; refusing to overwrite hand corrections (delete it first)")
    path.parent.mkdir(parents=True, exist_ok=True)
    width = max((len(t) for s in sentences for t in s["tokens"]), default=0) + 2
    lines = [HEADER.format(subset=subset, labels=" ".join(LABEL_LIST))]
    for s in sentences:
        lines.append(f"# id: {s['id']}\n# text: {' '.join(s['tokens'])}\n# reviewed: no")
        lines.extend(f"{tok:<{width}}\t{tag}" for tok, tag in zip(s["tokens"], s["tags"]))
        lines.append("")
    path.write_text("\n".join(lines))


def read_gold(path):
    """Parse a gold file. Raises ValueError with file:line for malformed lines or unknown labels."""
    sentences, current = [], None

    def finish():
        if current and current["tokens"]:
            sentences.append(current)

    for lineno, raw in enumerate(Path(path).read_text().splitlines(), start=1):
        line = raw.strip()
        if line.startswith("# id:"):
            finish()
            current = {"id": line[5:].strip(), "reviewed": False, "tokens": [], "tags": []}
        elif line.startswith("# reviewed:") and current is not None:
            current["reviewed"] = line.split(":", 1)[1].strip().lower() in ("yes", "y", "true")
        elif not line or line.startswith("#"):
            continue
        else:
            if current is None:
                raise ValueError(f"{path}:{lineno}: token line before any '# id:' header")
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"{path}:{lineno}: expected 'token<TAB>label', got {raw!r}")
            token, tag = parts
            if tag not in LABEL_LIST:
                raise ValueError(f"{path}:{lineno}: unknown label {tag!r} (valid: {' '.join(LABEL_LIST)})")
            current["tokens"].append(token)
            current["tags"].append(tag)
    finish()
    return sentences


def sample_nyt(data_path, n, seed=42):
    """Sample n sentences from the train_bert_colab.py test split, pre-labeled with fix_labels()."""
    test = split(dedupe(load_sentences(data_path)), seed=seed)["test"]
    picked = random.Random(seed + 1).sample(test, n)
    return [{"id": f"nyt-{i + 1:04d}", "tokens": tokens, "tags": fix_labels(tokens, tags)}
            for i, (tokens, tags) in enumerate(picked)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="subset", required=True)
    nyt = sub.add_parser("nyt", help="sample from the NYT held-out test split")
    nyt.add_argument("--data", default="data/training_data.json")
    nyt.add_argument("--n", type=int, default=50)
    nyt.add_argument("--seed", type=int, default=42, help="must match the training --seed")
    nyt.add_argument("--out", default="gold/nyt.tsv")
    args = parser.parse_args()

    if args.subset == "nyt":
        sentences = sample_nyt(args.data, args.n, args.seed)
    try:
        write_gold(args.out, sentences, args.subset)
    except FileExistsError as e:
        sys.exit(str(e))
    print(f"Wrote {len(sentences)} pre-labeled sentences to {args.out}")


if __name__ == "__main__":
    main()
