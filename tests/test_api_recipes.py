"""Endpoint tests for /api/recipes/*, with auth, the LLM, BERT and the encoder replaced by fakes."""
import os

import numpy as np
import pymysql
import pytest
from fastapi.testclient import TestClient

# api.py does its setup at import time. Keep it hermetic: no recipe loading, no Hub model,
# and no connection to a real MySQL server.
os.environ["RECIPES_CSV"] = ""
os.environ["BERT_MODEL_ID"] = ""
_real_connect = pymysql.connect


def _no_db(*args, **kwargs):
    raise pymysql.Error("database disabled in tests")


pymysql.connect = _no_db
try:
    import api
finally:
    pymysql.connect = _real_connect

RECIPES = [
    {"id": 1, "name": "Veggie Pasta", "minutes": 25, "tags": ["Vegan"], "ingredients": ["penne", "tomatoes", "basil"],
     "amount_hints": ["8", "2", None], "steps": []},
    {"id": 2, "name": "Mushroom Pasta", "minutes": 20, "tags": [], "ingredients": ["penne", "mushrooms", "butter"],
     "amount_hints": None, "steps": []},
    {"id": 3, "name": "Slow Ragu", "minutes": 180, "tags": [], "ingredients": ["penne", "beef", "tomatoes"],
     "amount_hints": None, "steps": []},
]


PASTA_PARSE = ('{"include_ingredients": [], "exclude_ingredients": ["mushrooms"], "diet": "vegetarian", '
               '"max_minutes": 30, "free_text": "pasta"}')


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "recipes", RECIPES)
    monkeypatch.setattr(api, "recipes_by_id", {r["id"]: r for r in RECIPES})
    monkeypatch.setattr(api, "recipe_embeddings", np.eye(3, dtype=np.float32))
    monkeypatch.setattr(api, "encode_query", lambda texts: np.array([[1, 0, 0]], dtype=np.float32))
    monkeypatch.setattr(api, "ask_llm", lambda prompt: PASTA_PARSE)
    monkeypatch.setattr(api, "nlp", None)  # use the stand-in parser
    substitute_calls = []

    def fake_substitute(item, amount, unit, constraint):
        substitute_calls.append((item, constraint))
        return {"found": True, "substitute_item": f"vegan {item}", "new_amount": amount, "new_unit": unit,
                "reason": "test"}

    monkeypatch.setattr(api, "get_substitute", fake_substitute)
    api.app.dependency_overrides[api.get_current_user] = lambda: {"id": 1, "email": "test@example.com"}
    c = TestClient(api.app)
    c.substitute_calls = substitute_calls
    yield c
    api.app.dependency_overrides.clear()


def test_search_parses_filters_and_returns_cards(client):
    response = client.post("/api/recipes/search", json={"query": "vegetarian pasta under 30 minutes, no mushrooms"})
    assert response.status_code == 200
    body = response.json()
    assert body["constraints"]["exclude_ingredients"] == ["mushrooms"]
    assert body["parser"] == "llm"
    assert body["unsupported_diet"] is None
    assert [r["id"] for r in body["results"]] == [1]  # 2 has mushrooms, 3 takes 180 min and has beef
    card = body["results"][0]
    assert card["top_ingredients"] == ["penne", "tomatoes", "basil"] and card["minutes"] == 25
    assert card["missing_ingredients"] == []


def test_search_requires_auth():
    api.app.dependency_overrides.clear()
    response = TestClient(api.app).post("/api/recipes/search", json={"query": "pasta"})
    assert response.status_code in (401, 403)


def test_search_rejects_empty_query(client):
    assert client.post("/api/recipes/search", json={"query": "  "}).status_code == 400


def test_search_without_recipes_is_503(client, monkeypatch):
    monkeypatch.setattr(api, "recipe_embeddings", None)
    assert client.post("/api/recipes/search", json={"query": "pasta"}).status_code == 503


def test_grocery_list_combines_food_com_and_pasted_recipe(client):
    response = client.post("/api/recipes/grocery-list", json={
        "recipe_ids": [1, 2],
        "constraints": {"diet": "vegetarian", "exclude_ingredients": []},
        "pasted_recipe": {"name": "My sauce", "text": "- 2 cups chopped tomatoes\n\n* 1 tablespoon butter\n"},
    })
    assert response.status_code == 200
    body = response.json()
    by_name = {i["name"]: i for i in body["items"]}
    assert by_name["tomato"]["display"] == "2 cups tomato + more for 1 other recipe (amount not given)"
    assert by_name["tomato"]["hints"] == [{"recipe": "Veggie Pasta", "amount": "2"}]
    assert by_name["penne"]["display"] == "penne (amount varies by recipe)"
    assert by_name["basil"]["display"] == "basil (amount not given)"
    assert body["parser"] == "fallback"
    assert body["substitutions"] == []  # nothing breaks "vegetarian"


def test_grocery_list_runs_substitutions_for_constraints(client):
    response = client.post("/api/recipes/grocery-list", json={
        "recipe_ids": [2], "constraints": {"diet": "vegan", "exclude_ingredients": ["mushrooms"]},
    })
    body = response.json()
    assert sorted(client.substitute_calls) == [("butter", "vegan"), ("mushroom", "no mushroom")]
    by_name = {i["name"]: i for i in body["items"]}
    assert by_name["butter"]["substitute"]["substitute_item"] == "vegan butter"


def test_grocery_list_pasted_only_works_without_recipe_data(client, monkeypatch):
    monkeypatch.setattr(api, "recipe_embeddings", None)
    response = client.post("/api/recipes/grocery-list", json={"pasted_recipe": {"text": "1 cup rice\n1/2 cup rice"}})
    assert response.status_code == 200
    assert [i["display"] for i in response.json()["items"]] == ["1 1/2 cups rice"]


@pytest.mark.parametrize("payload, status", [
    ({}, 400),
    ({"pasted_recipe": {"text": "\n  \n"}}, 400),
    ({"recipe_ids": [999]}, 404),
])
def test_grocery_list_errors(client, payload, status):
    assert client.post("/api/recipes/grocery-list", json=payload).status_code == status


def test_search_reports_fallback_parser(client, monkeypatch):
    def broken_llm(prompt):
        raise RuntimeError("API key not valid")

    monkeypatch.setattr(api, "ask_llm", broken_llm)
    body = client.post("/api/recipes/search", json={"query": "pasta under 30 minutes, no mushrooms"}).json()
    assert body["parser"] == "fallback"
    assert body["constraints"]["exclude_ingredients"] == ["mushrooms"] and body["constraints"]["max_minutes"] == 30
    assert [r["id"] for r in body["results"]] == [1]  # filters still applied


def test_gluten_free_oats_are_noted_in_search_and_grocery_list(client, monkeypatch):
    granola = {"id": 4, "name": "Granola", "minutes": 30, "tags": [], "ingredients": ["rolled oats", "honey"],
               "amount_hints": None, "steps": []}
    monkeypatch.setattr(api, "recipes", [granola])
    monkeypatch.setattr(api, "recipes_by_id", {4: granola})
    monkeypatch.setattr(api, "recipe_embeddings", np.eye(1, 3, dtype=np.float32))
    monkeypatch.setattr(api, "ask_llm", lambda prompt: '{"diet": "gluten-free", "free_text": "granola"}')

    card = client.post("/api/recipes/search", json={"query": "gluten-free granola"}).json()["results"][0]
    assert card["diet_notes"] == [{"ingredient": "rolled oats", "note": "use oats labeled gluten-free"}]

    body = client.post("/api/recipes/grocery-list", json={"recipe_ids": [4], "constraints": {"diet": "gluten-free"}}).json()
    by_name = {i["name"]: i for i in body["items"]}
    assert by_name["rolled oat"]["diet_note"] == "use oats labeled gluten-free"
    assert "diet_note" not in by_name["honey"]
    assert body["substitutions"] == []


def test_get_recipe_returns_ingredients_with_hints_and_steps(client, monkeypatch):
    recipe = {**RECIPES[0], "steps": ["Boil the penne.", "Toss with tomatoes and basil."]}
    monkeypatch.setattr(api, "recipes_by_id", {1: recipe, 2: RECIPES[1]})
    body = client.get("/api/recipes/1").json()
    assert body["name"] == "Veggie Pasta" and body["steps"] == recipe["steps"]
    assert body["ingredients"] == [{"name": "penne", "amount": "8"}, {"name": "tomatoes", "amount": "2"},
                                   {"name": "basil", "amount": None}]
    assert [i["amount"] for i in client.get("/api/recipes/2").json()["ingredients"]] == [None, None, None]
    assert client.get("/api/recipes/999").status_code == 404
