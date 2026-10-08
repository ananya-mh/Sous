import json

import numpy as np

from evaluate_search import QUERIES, evaluate, parse_matches, summarize
from recipe_search import QueryConstraints

MUSHROOM_PASTA = {"id": 1, "name": "Mushroom Pasta", "minutes": 20, "tags": [], "ingredients": ["penne", "mushrooms"],
                  "amount_hints": None, "steps": []}
ONES = np.ones((1, 2), dtype=np.float32)
ENCODE = lambda texts: np.ones((1, 2))


def test_parse_matches_normalizes():
    expected = QueryConstraints(diet="vegetarian", max_minutes=30, exclude_ingredients=["mushrooms"])
    parsed = QueryConstraints(diet="Vegetarian", max_minutes=30, exclude_ingredients=["Mushroom"])
    assert parse_matches(parsed, expected) == {"diet": True, "max_minutes": True, "exclude_ingredients": True}
    assert parse_matches(QueryConstraints(), expected) == {"diet": False, "max_minutes": False, "exclude_ingredients": False}


def failing_llm(prompt):
    raise RuntimeError("no key")


def free_text_llm(prompt):
    """A 'successful' parse that ignores every constraint."""
    query = prompt.split("Request: ", 1)[1].split("\n", 1)[0]
    return json.dumps({"free_text": query})


def test_wrong_parse_shows_up_as_violations():
    rows = evaluate([MUSHROOM_PASTA], ONES, ENCODE, "llm", free_text_llm)
    first = rows[0]  # "vegetarian pasta under 30 minutes, no mushrooms"
    assert first["parsed_by"] == "llm"
    assert first["results"][0]["violations"] == ["exclude_ingredients"]  # checked against hand-written constraints
    assert first["parse_matches"]["exclude_ingredients"] is False
    s = summarize(rows)
    assert s["queries"] == len(QUERIES) and s["results_returned"] == len(QUERIES)
    assert s["hard_constraints_satisfied"] < s["results_returned"]


def test_failed_llm_uses_fallback():
    # The fallback reads "no mushrooms", so the mushroom recipe is filtered out instead of returned.
    rows = evaluate([MUSHROOM_PASTA], ONES, ENCODE, "llm", failing_llm)
    first = rows[0]
    assert first["parsed_by"] == "fallback" and first["results"] == []
    assert first["parse_matches"] == {"diet": True, "max_minutes": True, "exclude_ingredients": True}
    assert summarize(rows)["fallback_used"] == len(QUERIES)


def test_fallback_mode_parses_without_llm():
    rows = evaluate([MUSHROOM_PASTA], ONES, ENCODE, "fallback")
    assert all(r["parsed_by"] == "fallback" for r in rows)
    assert summarize(rows)["queries_parsed_correctly"] is not None


def test_hand_written_mode_reports_no_parse_check():
    rows = evaluate([MUSHROOM_PASTA], ONES, ENCODE, "none")
    assert all(r["parse_matches"] is None for r in rows)
    assert summarize(rows)["queries_parsed_correctly"] is None


def test_load_queries_reads_file_and_heldout_set_is_valid(tmp_path):
    import pytest
    from evaluate_search import load_queries
    from pathlib import Path

    heldout = json.loads(Path("scripts/eval_queries_heldout.json").read_text())
    assert len(load_queries("scripts/eval_queries_heldout.json")) == len(heldout["queries"]) == 15
    example = heldout["_example"]
    path = tmp_path / "q.json"
    path.write_text(json.dumps({"queries": [example]}))
    [(query, expected)] = load_queries(path)
    assert query == example["query"] and expected.diet == "vegan" and expected.max_minutes == 60

    path.write_text(json.dumps({**heldout, "queries": []}))
    with pytest.raises(SystemExit, match="no queries yet"):
        load_queries(path)
    path.write_text(json.dumps({"queries": [{"query": "soup", "expected": {"exclude": ["onions"]}}]}))
    with pytest.raises(SystemExit, match=r"query 1: unknown expected field\(s\) \['exclude'\]"):
        load_queries(path)


def test_evaluate_uses_custom_queries():
    queries = [("pasta without mushrooms", QueryConstraints(free_text="pasta", exclude_ingredients=["mushrooms"])),
               ("mushroom pasta", QueryConstraints(include_ingredients=["mushrooms"]))]
    rows = evaluate([MUSHROOM_PASTA], ONES, ENCODE, "fallback", queries=queries)
    s = summarize(rows)
    assert s["queries"] == 2 and s["include_queries"] == 1 and s["include_all_present"] == 1
    assert s["queries_parsed_correctly"] == 2 and rows[0]["results"] == []
