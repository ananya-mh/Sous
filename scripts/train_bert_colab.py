"""Self-contained BERT ingredient-NER training script (local or Colab).

Fixes over train_BERT.py: deduplicates (case-insensitively) before splitting, splits
train/val/test 80/10/10, picks the best checkpoint on val (not test), and reports
entity-level P/R/F1 per label on test. `--fix-labels` cleans up rule-labeling noise.

Colab:
    !pip install -q -r requirements-train.txt
    !python scripts/train_bert_colab.py                 # original labels
    !python scripts/train_bert_colab.py --fix-labels    # fixed labels

Each run writes Results/metrics_<label_mode>.json and saves the model to
runs/<label_mode>/model (+ .zip), in the format api.py's pipeline() loads.
Smoke test on CPU: --max-samples 500 --epochs 1 --base-model prajjwal1/bert-tiny
"""
import argparse
import json
import random
import re
import shutil
from pathlib import Path

# Must match bert_recipe_model/config.json so the label ids stay compatible.
LABEL_LIST = ["O", "B-AMT", "I-AMT", "B-UNIT", "I-UNIT", "B-NAME", "I-NAME", "B-DESC", "I-DESC"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ENTITY_TYPES = ["AMT", "UNIT", "NAME", "DESC"]


# --- Data ---

def load_sentences(path):
    """Load preprocess.py output: [[[token, tag], ...], ...]. Drops sentences with unknown tags."""
    with open(path) as f:
        raw = json.load(f)
    sentences = []
    for sentence in raw:
        tokens = [t for t, _ in sentence]
        tags = [g for _, g in sentence]
        if tokens and all(g in LABEL2ID for g in tags):
            sentences.append((tokens, tags))
    return sentences


def dedupe(sentences):
    """Keep the first copy of each token sequence. Case-insensitive: the model is uncased."""
    seen, unique = set(), []
    for tokens, tags in sentences:
        key = tuple(t.lower() for t in tokens)
        if key not in seen:
            seen.add(key)
            unique.append((tokens, tags))
    return unique


def split(sentences, seed=42, ratios=(0.8, 0.1, 0.1)):
    """Deterministic shuffle + slice. Splits depend only on tokens, so both label modes match."""
    order = list(range(len(sentences)))
    random.Random(seed).shuffle(order)
    n_train = int(len(order) * ratios[0])
    n_val = int(len(order) * ratios[1])
    pick = lambda idx: [sentences[i] for i in idx]
    return {
        "train": pick(order[:n_train]),
        "val": pick(order[n_train:n_train + n_val]),
        "test": pick(order[n_train + n_val:]),
    }


# --- Label fixes ---

PUNCT_CHARS = set(",.;:()[]{}'’‘\"-–—!?*")
FUNCTION_WORDS = {"a", "an", "the", "of", "de", "and", "or", "in", "as", "with", "for", "to", "&"}
RANGE_WORDS = {"-", "–", "to", "or"}
FLUID_WORDS = {"fluid", "fl", "fl."}
_FRACTION = re.compile(r"^(\d+/\d+|[½⅓⅔¼¾⅕⅛⅜⅝⅞])$")


def _type(tag):
    return None if tag == "O" else tag[2:]


def fix_labels(tokens, tags):
    """Repair the rule labels from preprocess.py.

    - punctuation is never part of an entity
    - function words ("a", "of", ...) are NAME only inside a name ("cream of tartar"), else O
    - mixed numbers / ranges become one AMT span ("1 1/2", "2 to 3"); "fluid ounces" one UNIT span
    - consecutive same-type tokens continue the span with I- tags (preprocess never emits I-DESC)
    """
    tags = list(tags)
    n = len(tokens)
    low = [t.lower() for t in tokens]

    for i, tok in enumerate(tokens):
        if tok and all(c in PUNCT_CHARS for c in tok):
            tags[i] = "O"

    for i in range(n):
        if low[i] in FUNCTION_WORDS and _type(tags[i]) == "NAME":
            inside = (i > 0 and _type(tags[i - 1]) == "NAME"
                      and i + 1 < n and _type(tags[i + 1]) == "NAME" and low[i + 1] not in FUNCTION_WORDS)
            tags[i] = "I-NAME" if inside else "O"

    for i in range(1, n):
        if _type(tags[i]) == "AMT" and _type(tags[i - 1]) == "AMT" and _FRACTION.match(tokens[i]) \
                and tokens[i - 1].isdigit():
            tags[i] = "I-AMT"  # "1 1/2"
        if low[i] in RANGE_WORDS and i + 1 < n and _type(tags[i - 1]) == "AMT" and _type(tags[i + 1]) == "AMT":
            tags[i] = tags[i + 1] = "I-AMT"  # "2 to 3"
        if _type(tags[i]) == "UNIT" and low[i - 1] in FLUID_WORDS:
            tags[i - 1], tags[i] = "B-UNIT", "I-UNIT"  # "fluid ounces"

    for i in range(1, n):
        if tags[i] == "B-DESC" and _type(tags[i - 1]) == "DESC":
            tags[i] = "I-DESC"

    for i in range(n):  # valid BIO: an I- tag must follow the same type
        if tags[i].startswith("I-") and (i == 0 or _type(tags[i - 1]) != _type(tags[i])):
            tags[i] = "B-" + tags[i][2:]
    return tags


def apply_label_mode(sentences, fixed):
    return [(tokens, fix_labels(tokens, tags) if fixed else list(tags)) for tokens, tags in sentences]


# --- Metrics ---

def get_spans(tags):
    """BIO tags -> [(type, start, end)], conlleval rules: B- or a type change starts a new span."""
    spans, start, current = [], 0, None
    for i, tag in enumerate(list(tags) + ["O"]):
        prefix, etype = (tag[0], tag[2:]) if tag != "O" else ("O", None)
        if current is not None and (prefix != "I" or etype != current):
            spans.append((current, start, i - 1))
            current = None
        if prefix in ("B", "I") and current is None:
            start, current = i, etype
    return spans


def _prf(tp, n_pred, n_true):
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_true if n_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1-score": f1, "support": float(n_true)}


def entity_report(y_true, y_pred):
    """Entity-level (exact span + type) P/R/F1 per label, plus micro/macro/weighted averages.

    Same numbers and keys as seqeval's classification_report(output_dict=True, zero_division=0);
    implemented here because seqeval only ships an sdist, which can fail to build on Colab.
    """
    true = {(i, *s) for i, tags in enumerate(y_true) for s in get_spans(tags)}
    pred = {(i, *s) for i, tags in enumerate(y_pred) for s in get_spans(tags)}
    report = {}
    for etype in sorted({s[1] for s in true | pred}):
        t = {s for s in true if s[1] == etype}
        p = {s for s in pred if s[1] == etype}
        report[etype] = _prf(len(t & p), len(p), len(t))
    per_type = list(report.values())
    total = sum(r["support"] for r in per_type)
    report["micro avg"] = _prf(len(true & pred), len(pred), len(true))
    report["macro avg"] = {m: sum(r[m] for r in per_type) / len(per_type) if per_type else 0.0
                           for m in ("precision", "recall", "f1-score")}
    report["weighted avg"] = {m: sum(r[m] * r["support"] for r in per_type) / total if total else 0.0
                              for m in ("precision", "recall", "f1-score")}
    report["macro avg"]["support"] = report["weighted avg"]["support"] = float(total)
    return report


def token_report(y_true, y_pred):
    """Per-type token-level P/R/F1 with B/I collapsed: insensitive to span conventions."""
    out = {}
    pairs = [(_type(t), _type(p)) for ts, ps in zip(y_true, y_pred) for t, p in zip(ts, ps)]
    for etype in ENTITY_TYPES:
        tp = sum(1 for t, p in pairs if t == etype and p == etype)
        fp = sum(1 for t, p in pairs if t != etype and p == etype)
        fn = sum(1 for t, p in pairs if t == etype and p != etype)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[etype] = {"precision": precision, "recall": recall, "f1-score": f1, "support": tp + fn}
    return out


# --- Model helpers (heavy imports stay inside functions so tests stay light) ---

def tokenize_and_align(tokenizer, sentences):
    """Label only the first subword of each word; the rest get -100 (as in train_BERT.py)."""
    features = []
    for start in range(0, len(sentences), 1000):
        batch = sentences[start:start + 1000]
        enc = tokenizer([t for t, _ in batch], is_split_into_words=True, truncation=True)
        for i, (_, tags) in enumerate(batch):
            previous, labels = None, []
            for word_idx in enc.word_ids(batch_index=i):
                labels.append(-100 if word_idx is None or word_idx == previous else LABEL2ID[tags[word_idx]])
                previous = word_idx
            features.append({"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i],
                             "labels": labels})
    return features


def predict_tags(model, tokenizer, token_lists, batch_size=64):
    """Predict one tag per input word (first-subword prediction); truncated words get 'O'."""
    import torch
    model.eval()
    device = next(model.parameters()).device
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    predictions = []
    for start in range(0, len(token_lists), batch_size):
        batch = token_lists[start:start + batch_size]
        enc = tokenizer(batch, is_split_into_words=True, truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
        best = logits.argmax(-1).cpu().tolist()
        for i, tokens in enumerate(batch):
            tags, previous = ["O"] * len(tokens), None
            for pos, word_idx in enumerate(enc.word_ids(batch_index=i)):
                if word_idx is not None and word_idx != previous:
                    tags[word_idx] = id2label[best[i][pos]]
                previous = word_idx
            predictions.append(tags)
    return predictions


def parse_like_api(nlp, text):
    """Same aggregation as parse_recipe_bert in api.py, for the post-save smoke test."""
    parsed = {"amount": [], "unit": [], "item": [], "descriptor": []}
    keys = {"AMT": "amount", "UNIT": "unit", "NAME": "item", "DESC": "descriptor"}
    for entity in nlp(text):
        if entity["entity_group"] in keys:
            parsed[keys[entity["entity_group"]]].append(entity["word"].replace("##", "").strip())
    return {k: " ".join(v) for k, v in parsed.items()}


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/training_data.json")
    parser.add_argument("--fix-labels", action="store_true", help="apply fix_labels() to every split")
    parser.add_argument("--base-model", default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None, help="cap train size (smoke tests)")
    parser.add_argument("--output-dir", default=None, help="default: runs/<label_mode>/model, so runs never overwrite each other")
    parser.add_argument("--results-dir", default="Results")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()

    import numpy as np
    import torch
    from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                              EarlyStoppingCallback, Trainer, TrainingArguments, pipeline, set_seed)

    set_seed(args.seed)
    label_mode = "fixed" if args.fix_labels else "original"
    run_tag = label_mode + ("_smoke" if args.max_samples else "")
    output_dir = Path(args.output_dir or f"runs/{run_tag}/model")

    raw = load_sentences(args.data)
    unique = dedupe(raw)
    splits_original = split(unique, seed=args.seed)
    if args.max_samples:
        cap = {"train": args.max_samples, "val": max(args.max_samples // 8, 1), "test": max(args.max_samples // 8, 1)}
        splits_original = {k: v[:cap[k]] for k, v in splits_original.items()}
    splits = {k: apply_label_mode(v, args.fix_labels) for k, v in splits_original.items()}
    changed = sum(1 for (_, a), (_, b) in zip(splits_original["train"], splits["train"]) if a != b)
    print(f"Sentences: {len(raw)} raw, {len(unique)} unique. Split sizes: "
          + ", ".join(f"{k}={len(v)}" for k, v in splits.items()) + f". Label mode: {label_mode}"
          + (f" ({changed} train sentences changed)" if args.fix_labels else ""))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    train_features = tokenize_and_align(tokenizer, splits["train"])
    val_features = tokenize_and_align(tokenizer, splits["val"])

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        y_true = [[LABEL_LIST[l] for pr, l in zip(ps, ls) if l != -100] for ps, ls in zip(predictions, labels)]
        y_pred = [[LABEL_LIST[pr] for pr, l in zip(ps, ls) if l != -100] for ps, ls in zip(predictions, labels)]
        return {"f1": entity_report(y_true, y_pred)["micro avg"]["f1-score"]}

    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model, num_labels=len(LABEL_LIST),
        id2label=dict(enumerate(LABEL_LIST)), label2id=LABEL2ID,
    )
    checkpoint_dir = output_dir.with_name(output_dir.name + "_checkpoints")  # per run, outside the zipped model
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_features,
        eval_dataset=val_features,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    # Test once, against both label versions of the same test sentences: each run is
    # scored on its own labels, and on the other mode's labels for a rough cross-check.
    test_tokens = [t for t, _ in splits["test"]]
    predicted = predict_tags(trainer.model, tokenizer, test_tokens)
    references = {
        "original": [tags for _, tags in splits_original["test"]],
        "fixed": [tags for _, tags in apply_label_mode(splits_original["test"], True)],
    }
    metrics = {
        "label_mode": label_mode,
        "base_model": args.base_model,
        "hyperparameters": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                            "weight_decay": 0.01, "seed": args.seed, "early_stopping_patience": 2},
        "data": {"sentences_raw": len(raw), "sentences_unique": len(unique),
                 **{f"{k}_size": len(v) for k, v in splits.items()},
                 "train_sentences_relabeled": changed if args.fix_labels else 0,
                 "max_samples": args.max_samples},
        "best_val_entity_f1": trainer.state.best_metric,
        "test": {
            f"vs_{ref}_labels": {"entity_level": entity_report(refs, predicted),
                                 "token_level": token_report(refs, predicted)}
            for ref, refs in references.items()
        },
        "note": "Test labels are rule-generated; use evaluate_gold.py on the hand-corrected gold set "
                "to compare runs against human labels.",
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / f"metrics_{run_tag}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    shutil.rmtree(checkpoint_dir, ignore_errors=True)

    own = metrics["test"][f"vs_{label_mode}_labels"]["entity_level"]
    print(f"\nTest entity-level F1 vs {label_mode} labels:")
    for etype in ENTITY_TYPES + ["micro avg"]:
        if etype in own:
            r = own[etype]
            print(f"  {etype:<10} P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1-score']:.4f} n={int(r['support'])}")
    print(f"Metrics: {metrics_path}")

    # Smoke test: load exactly the way api.py does.
    nlp = pipeline("token-classification", model=str(output_dir), aggregation_strategy="first")
    for line in ["1 cup chopped tomatoes", "2 1/2 cups all-purpose flour", "3 cloves garlic, minced"]:
        print(f"  {line!r} -> {parse_like_api(nlp, line)}")

    if not args.no_zip:
        archive = shutil.make_archive(str(output_dir), "zip", str(output_dir))
        print(f"Model: {output_dir} (zipped: {archive})")


if __name__ == "__main__":
    main()
