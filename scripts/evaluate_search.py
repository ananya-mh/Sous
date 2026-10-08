"""Evaluate recipe search on 10 hand-written queries (or a JSON query file via --queries).

For each query, the top 5 results are checked against hand-written expected constraints:
  - hard constraints: max_minutes, excluded ingredients, diet (same checks as the search filter)
  - included ingredients: does the result contain every requested ingredient?

    python scripts/evaluate_search.py                    # the LLM (Groq) parses each query (needs GROQ_API_KEY)
    python scripts/evaluate_search.py --fallback-parser  # the regex fallback parses each query
    python scripts/evaluate_search.py --no-llm           # search with the hand-written constraints
    python scripts/evaluate_search.py --fallback-parser --queries scripts/eval_queries_heldout.json

The built-in queries were written alongside the fallback parser's patterns, so its score on them is
optimistic; scripts/eval_queries_heldout.json holds queries written without looking at the patterns.

With a parser (the LLM or the fallback), a hard-constraint failure means the query was parsed wrong (the filter itself
can't let a violation through). With --no-llm the hard-constraint rate is 100% by
construction, so that mode is mainly useful for the included-ingredient rate.
Uses RECIPES_CSV / RECIPES_LIMIT and the same caches as api.py.
"""
import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from grocery_list import canonical_diet, normalize_name
from recipe_search import (DEFAULT_GROQ_MODEL, QueryConstraints, fallback_parse, groq_completer, load_or_build_embeddings, load_recipes, parse_query,
                           recipe_contains, search, sentence_encoder, violations)

QUERIES = [
    ("vegetarian pasta under 30 minutes, no mushrooms",
     QueryConstraints(free_text="pasta", diet="vegetarian", max_minutes=30, exclude_ingredients=["mushrooms"])),
    ("quick vegan curry in under 45 minutes", QueryConstraints(free_text="curry", diet="vegan", max_minutes=45)),
    ("gluten-free chocolate dessert", QueryConstraints(free_text="chocolate dessert", diet="gluten-free")),
    ("chicken dinner with no dairy",
     QueryConstraints(free_text="dinner", include_ingredients=["chicken"], diet="dairy-free")),
    ("low-carb breakfast with eggs under 20 minutes",
     QueryConstraints(free_text="breakfast", include_ingredients=["eggs"], diet="low-carb", max_minutes=20)),
    ("nut-free cookies", QueryConstraints(free_text="cookies", diet="nut-free")),
    ("soup with lentils and carrots, no onions",
     QueryConstraints(free_text="soup", include_ingredients=["lentils", "carrots"], exclude_ingredients=["onions"])),
    ("salmon under 40 minutes without garlic",
     QueryConstraints(include_ingredients=["salmon"], max_minutes=40, exclude_ingredients=["garlic"])),
    ("vegetarian chili with black beans",
     QueryConstraints(free_text="chili", include_ingredients=["black beans"], diet="vegetarian")),
    ("dairy-free banana smoothie in 10 minutes",
     QueryConstraints(free_text="smoothie", include_ingredients=["banana"], diet="dairy-free", max_minutes=10)),
]


EXPECTED_FIELDS = set(QueryConstraints.model_fields)


def load_queries(path):
    """[(query, QueryConstraints)] from a JSON file: {"queries": [{"query": ..., "expected": {...}}, ...]}."""
    data = json.loads(Path(path).read_text())
    entries = data.get("queries", []) if isinstance(data, dict) else data
    if not entries:
        raise SystemExit(f"{path} has no queries yet: add entries to its \"queries\" list (see \"_example\")")
    queries = []
    for n, entry in enumerate(entries, 1):
        query, expected = entry.get("query", "").strip(), entry.get("expected", {})
        unknown = set(expected) - EXPECTED_FIELDS
        if not query or unknown:
            problem = "empty \"query\"" if not query else f"unknown expected field(s) {sorted(unknown)}"
            raise SystemExit(f"{path}: query {n}: {problem}; fields are {sorted(EXPECTED_FIELDS)}")
        queries.append((query, QueryConstraints(**expected)))
    return queries


def parse_matches(parsed, expected):
    """Which hard-constraint fields the parser got right ("Vegetarian" == "vegetarian", "mushroom" == "mushrooms")."""
    names = lambda xs: {normalize_name(x) for x in xs}
    return {
        "diet": canonical_diet(parsed.diet) == canonical_diet(expected.diet),
        "max_minutes": parsed.max_minutes == expected.max_minutes,
        "exclude_ingredients": names(parsed.exclude_ingredients) == names(expected.exclude_ingredients),
    }


def evaluate(recipes, embeddings, encode, parser="none", ask_llm=None, k=5, queries=QUERIES, llm_delay=0.0):
    """parser: "llm", "fallback", or "none" (use the hand-written constraints).
    ask_llm: prompt -> reply text. llm_delay: seconds between LLM calls, to stay under free-tier rate limits."""
    rows = []
    for n, (query, expected) in enumerate(queries):
        used = parser
        if parser == "llm":
            if n and llm_delay:
                time.sleep(llm_delay)
            constraints, used = parse_query(query, ask_llm)  # "fallback" if the call failed
        elif parser == "fallback":
            constraints = fallback_parse(query)
        else:
            constraints = expected
        results = search(constraints, query, recipes, embeddings, encode, k=k)
        checked = []
        for r in results:
            recipe = r["recipe"]
            checked.append({
                "id": recipe["id"], "name": recipe["name"], "minutes": recipe["minutes"],
                "violations": violations(recipe, expected),  # against the hand-written constraints
                "missing_ingredients": [w for w in expected.include_ingredients if not recipe_contains(recipe, w)],
                "diet_notes": [n["ingredient"] for n in r["diet_notes"]],
            })
        rows.append({
            "query": query,
            "expected": expected.model_dump(),
            "parsed": constraints.model_dump(),
            "parsed_by": used,
            "parse_matches": parse_matches(constraints, expected) if parser != "none" else None,
            "candidates": sum(not violations(r, expected) for r in recipes),
            "results": checked,
        })
    return rows


def summarize(rows, k=5):
    parsed = [r for r in rows if r["parse_matches"] is not None]
    slots = len(rows) * k
    returned = sum(len(r["results"]) for r in rows)
    satisfied = sum(not x["violations"] for r in rows for x in r["results"])
    with_includes = [r for r in rows if r["expected"]["include_ingredients"]]
    include_returned = sum(len(r["results"]) for r in with_includes)
    include_all = sum(not x["missing_ingredients"] for r in with_includes for x in r["results"])
    return {
        "queries": len(rows),
        "results_returned": returned,
        "result_slots": slots,
        "hard_constraints_satisfied": satisfied,
        "hard_constraint_rate": satisfied / returned if returned else 0.0,
        "include_queries": len(with_includes),
        "include_results_returned": include_returned,
        "include_all_present": include_all,
        "include_all_rate": include_all / include_returned if include_returned else 0.0,
        "queries_parsed_correctly": sum(all(r["parse_matches"].values()) for r in parsed) if parsed else None,
        "fallback_used": sum(r["parsed_by"] == "fallback" for r in rows),
    }


def print_report(rows, summary, parser):
    for r in rows:
        ok = sum(not x["violations"] for x in r["results"])
        line = f"\n{r['query']!r}: {ok}/{len(r['results'])} satisfy all hard constraints ({r['candidates']} candidates)"
        if r["parse_matches"] is not None:
            wrong = [f for f, good in r["parse_matches"].items() if not good]
            line += f"; {r['parsed_by']} parse " + ("OK" if not wrong else f"wrong on {', '.join(wrong)}")
        print(line)
        for x in r["results"]:
            flags = []
            if x["violations"]:
                flags.append("VIOLATES " + ", ".join(x["violations"]))
            if x["missing_ingredients"]:
                flags.append("missing " + ", ".join(x["missing_ingredients"]))
            if x["diet_notes"]:
                flags.append("note: " + ", ".join(x["diet_notes"]))
            print(f"  {x['name'][:50]:<50} {x['minutes']:>4} min  {'; '.join(flags)}")

    s = summary
    print(f"\nTop-5 results satisfying every hard constraint: {s['hard_constraints_satisfied']}/{s['results_returned']} "
          f"({s['hard_constraint_rate']:.0%}); {s['results_returned']}/{s['result_slots']} slots filled")
    print(f"Top-5 results containing all requested ingredients ({s['include_queries']} queries ask for some): "
          f"{s['include_all_present']}/{s['include_results_returned']} ({s['include_all_rate']:.0%})")
    if parser == "none":
        print("(--no-llm: hard constraints are 100% by construction; run with a parser to test query parsing)")
    else:
        print(f"Queries parsed correctly (diet, max_minutes, exclusions): {s['queries_parsed_correctly']}/{s['queries']}"
              + (f"; the LLM failed and the fallback was used for {s['fallback_used']}" if parser == "llm" and s["fallback_used"] else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--no-llm", action="store_true", help="use the hand-written constraints instead")
    mode.add_argument("--fallback-parser", action="store_true", help="parse queries with the regex fallback")
    parser.add_argument("--queries", default=None,
                        help="JSON query file (e.g. scripts/eval_queries_heldout.json); default: the 10 built-in queries")
    parser.add_argument("--llm-delay", type=float, default=2.0,
                        help="seconds between LLM calls, for free-tier rate limits (default 2)")
    parser.add_argument("--out", default=None, help="default: Results/search_eval_<mode>[_<query file>].json")
    args = parser.parse_args()
    queries = load_queries(args.queries) if args.queries else QUERIES  # fail fast, before loading models

    load_dotenv()
    csv_path = os.getenv("RECIPES_CSV")
    if not csv_path:
        raise SystemExit("Set RECIPES_CSV (e.g. data/recipes.csv)")
    recipes = load_recipes(csv_path, int(os.getenv("RECIPES_LIMIT", "5000")),
                           cache_path=os.getenv("RECIPES_SUBSET_CACHE", "data/recipes_subset.json"))
    encode = sentence_encoder()
    embeddings = load_or_build_embeddings(
        recipes, os.getenv("RECIPES_EMBEDDINGS_CACHE", "data/recipe_embeddings.npz"), encode)

    parser_mode = "none" if args.no_llm else "fallback" if args.fallback_parser else "llm"
    ask_llm = None
    if parser_mode == "llm":
        ask_llm = groq_completer(os.getenv("GROQ_API_KEY"), os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL))
        if ask_llm is None:
            raise SystemExit("GROQ_API_KEY is not set; use --no-llm or --fallback-parser to skip LLM parsing")

    rows = evaluate(recipes, embeddings, encode, parser_mode, ask_llm, queries=queries, llm_delay=args.llm_delay)
    summary = summarize(rows)
    print_report(rows, summary, parser_mode)
    if parser_mode == "llm" and summary["fallback_used"] == len(rows):
        raise SystemExit("\nThe LLM failed on every query (see the warnings above), so nothing was written. "
                         "Check GROQ_API_KEY and GROQ_MODEL.")
    suffix = "_" + Path(args.queries).stem.removeprefix("eval_queries_") if args.queries else ""
    out = args.out or f"Results/search_eval_{parser_mode if parser_mode != 'none' else 'nollm'}{suffix}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"parser": parser_mode, "llm_model": os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL) if ask_llm else None,
                                          "query_file": args.queries, "summary": summary,
                                          "queries": rows}, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
