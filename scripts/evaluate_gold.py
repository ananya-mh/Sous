"""Score an NER model against the hand-labeled gold set, per subset (one file = one subset).

    python scripts/evaluate_gold.py --model runs/original/model --tag original
    python scripts/evaluate_gold.py --model runs/fixed/model --tag fixed
    python scripts/evaluate_gold.py --model your-user/sous-bert --tag hub

Reports entity-level (exact span + type) P/R/F1 per label, plus token-level per type, which
doesn't penalize span conventions (e.g. a model trained without I-DESC splitting
"finely chopped" into two spans). Writes Results/gold_metrics_<tag>.json.
"""
import argparse
import json
from pathlib import Path

from gold_set import read_gold
from train_bert_colab import ENTITY_TYPES, entity_report, predict_tags, token_report


def score(gold, predictions):
    y_true = [s["tags"] for s in gold]
    return {
        "sentences": len(gold),
        "reviewed": sum(s["reviewed"] for s in gold),
        "entity_level": entity_report(y_true, predictions),
        "token_level": token_report(y_true, predictions),
    }


def print_table(subset, result):
    print(f"\n{subset}: {result['sentences']} sentences ({result['reviewed']} reviewed)")
    print(f"  {'label':<10} {'P':>6} {'R':>6} {'F1':>6} {'n':>5}   token-F1")
    ent, tok = result["entity_level"], result["token_level"]
    for label in ENTITY_TYPES + ["micro avg", "macro avg"]:
        if label in ent:
            r = ent[label]
            tok_f1 = f"{tok[label]['f1-score']:.3f}" if label in tok else ""
            print(f"  {label:<10} {r['precision']:6.3f} {r['recall']:6.3f} {r['f1-score']:6.3f} "
                  f"{int(r['support']):5d}   {tok_f1}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="local folder or Hugging Face Hub id")
    parser.add_argument("--gold", nargs="+", default=None, help="gold files (default: gold/*.tsv)")
    parser.add_argument("--tag", default=None, help="name for the results file (default: model folder name)")
    parser.add_argument("--reviewed-only", action="store_true", help="skip sentences not marked reviewed")
    parser.add_argument("--results-dir", default="Results")
    args = parser.parse_args()

    from transformers import AutoModelForTokenClassification, AutoTokenizer

    gold_files = [Path(p) for p in args.gold] if args.gold else sorted(Path("gold").glob("*.tsv"))
    if not gold_files:
        raise SystemExit("No gold files found (expected gold/*.tsv). Run scripts/gold_set.py first.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model)

    results = {"model": args.model, "subsets": {}}
    for path in gold_files:
        gold = read_gold(path)
        if args.reviewed_only:
            gold = [s for s in gold if s["reviewed"]]
        if not gold:
            print(f"\n{path.stem}: no sentences to score, skipping")
            continue
        predictions = predict_tags(model, tokenizer, [s["tokens"] for s in gold])
        results["subsets"][path.stem] = score(gold, predictions)
        print_table(path.stem, results["subsets"][path.stem])
        unreviewed = len(gold) - results["subsets"][path.stem]["reviewed"]
        if unreviewed:
            print(f"  warning: {unreviewed} sentences not reviewed yet; these scores partly measure the rules")

    tag = args.tag or Path(args.model.rstrip("/")).name
    out = Path(args.results_dir) / f"gold_metrics_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
