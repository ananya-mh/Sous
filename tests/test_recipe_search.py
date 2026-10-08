import csv
import re

import numpy as np
import pytest

from recipe_search import (
    QueryConstraints, fallback_parse, iso_minutes, load_or_build_embeddings, load_recipes, parse_query, parse_r_vector,
    ranking_text, row_to_recipe, satisfies_diet, search, unsupported_diet, violations,
)


def make_recipe(id, name, ingredients, minutes=20, tags=()):
    return {"id": id, "name": name, "minutes": minutes, "tags": list(tags), "ingredients": ingredients,
            "amount_hints": None, "steps": []}


# --- CSV parsing ---

@pytest.mark.parametrize("text, expected", [
    ('c("4", "1/4", NA)', ["4", "1/4", None]),
    ('"powdered sugar"', ["powdered sugar"]),  # one-element vector
    ("NA", [None]),
    ("character(0)", []),
    ("", []),
    ('c("say \\"cheese\\"", "x")', ['say "cheese"', "x"]),
    ('c("Preheat.\\r\\nStir.")', ["Preheat.\nStir."]),
    ('c("Mac &amp; &quot;Cheese&quot;")', ['Mac & "Cheese"']),
])
def test_parse_r_vector(text, expected):
    assert parse_r_vector(text) == expected


@pytest.mark.parametrize("duration, minutes", [
    ("PT45M", 45), ("PT24H45M", 1485), ("PT1H", 60), ("P1DT2H", 1560), ("PT30S", 1), ("", None), ("junk", None),
])
def test_iso_minutes(duration, minutes):
    assert iso_minutes(duration) == minutes


def csv_row(**overrides):
    row = {"RecipeId": "38.0", "Name": "Berry Dessert", "TotalTime": "PT45M", "RecipeCategory": "Frozen Desserts",
           "Keywords": 'c("Dessert", "Healthy")', "RecipeIngredientQuantities": 'c("4", "1⁄4")',
           "RecipeIngredientParts": 'c("blueberries", "sugar")', "RecipeInstructions": 'c("Mix.", "Freeze.")'}
    row.update(overrides)
    return row


def test_row_to_recipe_aligned():
    r = row_to_recipe(csv_row())
    assert r == {"id": 38, "name": "Berry Dessert", "minutes": 45, "tags": ["Dessert", "Healthy", "Frozen Desserts"],
                 "ingredients": ["blueberries", "sugar"], "amount_hints": ["4", "1/4"], "steps": ["Mix.", "Freeze."]}


def test_row_to_recipe_skips_recipes_with_dropped_names():
    # 3 quantities, 2 names: Food.com dropped an ingredient name, so the list is incomplete.
    assert row_to_recipe(csv_row(RecipeIngredientQuantities='c("4", "1/4", "2")')) is None


def test_row_to_recipe_more_names_than_quantities_is_kept_without_hints():
    r = row_to_recipe(csv_row(RecipeIngredientQuantities='"4"'))
    assert r["ingredients"] == ["blueberries", "sugar"] and r["amount_hints"] is None


def test_row_to_recipe_keeps_na_hint_as_none():
    assert row_to_recipe(csv_row(RecipeIngredientQuantities='c("4", NA)'))["amount_hints"] == ["4", None]


@pytest.mark.parametrize("override", [{"RecipeIngredientParts": "character(0)"}, {"TotalTime": ""}])
def test_row_to_recipe_skips_unusable_rows(override):
    assert row_to_recipe(csv_row(**override)) is None


def write_csv(path, n):
    rows = [csv_row(RecipeId=f"{i}.0", Name=f"Recipe {i}") for i in range(n)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_load_recipes_samples_deterministically_and_caches(tmp_path):
    path, cache = tmp_path / "recipes.csv", tmp_path / "subset.json"
    write_csv(path, 50)
    first = load_recipes(path, limit=10, seed=1, cache_path=cache)
    assert len(first) == 10 and [r["id"] for r in first] == sorted(r["id"] for r in first)
    assert load_recipes(path, limit=10, seed=1) == first  # same sample without the cache
    assert load_recipes(path, limit=10, seed=2) != first
    cache.write_text(cache.read_text().replace("Recipe", "Cached"))
    assert load_recipes(path, limit=10, seed=1, cache_path=cache)[0]["name"].startswith("Cached")  # cache hit
    assert load_recipes(path, limit=5, seed=1, cache_path=cache)[0]["name"].startswith("Recipe")  # key changed


# --- Embeddings cache ---

def test_embeddings_cache_reuses_and_invalidates(tmp_path):
    calls = []

    def encode(texts):
        calls.append(len(texts))
        return np.ones((len(texts), 3), dtype=np.float32)

    cache = tmp_path / "emb.npz"
    recipes = [make_recipe(1, "a", ["x"]), make_recipe(2, "b", ["y"])]
    assert load_or_build_embeddings(recipes, cache, encode).shape == (2, 3)
    load_or_build_embeddings(recipes, cache, encode)
    assert calls == [2]  # second call hit the cache
    load_or_build_embeddings(recipes[:1], cache, encode)
    assert calls == [2, 1]  # different recipe ids: rebuilt
    load_or_build_embeddings([make_recipe(1, "renamed", ["x"])], cache, encode)
    assert calls == [2, 1, 1]  # same id, different text: rebuilt


# --- Query parsing ---

def fake_llm(text=None, error=None):
    def ask_llm(prompt):
        if error:
            raise error
        return text
    return ask_llm


def test_parse_query_reads_fenced_json():
    model = fake_llm('```json\n{"include_ingredients": [], "exclude_ingredients": ["mushrooms"], '
                     '"diet": "vegetarian", "max_minutes": 30, "free_text": "pasta"}\n```')
    c, parser = parse_query("vegetarian pasta under 30 minutes, no mushrooms", model)
    assert c == QueryConstraints(exclude_ingredients=["mushrooms"], diet="vegetarian", max_minutes=30, free_text="pasta")
    assert parser == "llm"


@pytest.mark.parametrize("model", [
    fake_llm(error=RuntimeError("API down")),
    fake_llm("not json"),
    fake_llm('{"max_minutes": "soon"}'),  # wrong type
    None,  # no GROQ_API_KEY
])
def test_parse_query_uses_fallback_parser_on_failure(model):
    c, parser = parse_query("quick soup under 20 minutes, no onions", model)
    assert parser == "fallback"
    assert c == QueryConstraints(max_minutes=20, exclude_ingredients=["onions"], free_text="quick soup")


def test_constraints_clean_up_values():
    c = QueryConstraints(include_ingredients=["", " basil "], max_minutes=0)
    assert c.include_ingredients == ["basil"] and c.max_minutes is None


# --- Filtering ---

PASTA = make_recipe(1, "Veggie Pasta", ["penne", "tomatoes", "basil"], minutes=25)
MUSHROOM_PASTA = make_recipe(2, "Mushroom Pasta", ["penne", "cremini mushrooms"], minutes=20)
SLOW_PASTA = make_recipe(3, "Slow Ragu", ["penne", "tomatoes"], minutes=180)
BACON_PASTA = make_recipe(4, "Carbonara", ["spaghetti", "bacon", "eggs"], minutes=25)


def test_violations_per_constraint():
    c = QueryConstraints(max_minutes=30, exclude_ingredients=["mushrooms"], diet="vegetarian")
    assert violations(PASTA, c) == []
    assert violations(MUSHROOM_PASTA, c) == ["exclude_ingredients"]
    assert violations(SLOW_PASTA, c) == ["max_minutes"]
    assert violations(BACON_PASTA, c) == ["diet"]


@pytest.mark.parametrize("recipe, diet, expected", [
    (make_recipe(1, "x", ["tofu"], tags=["Vegan"]), "vegan", True),
    (make_recipe(1, "x", ["tofu"]), "vegan", False),  # tag diets need the tag
    (make_recipe(1, "x", ["tofu", "honey"], tags=["Vegan"]), "vegan", False),  # ...and clean ingredients
    (make_recipe(1, "x", ["steak"], tags=["Very Low Carbs"]), "keto", True),
    (make_recipe(1, "x", ["steak", "potatoes"], tags=["Very Low Carbs"]), "low-carb", False),
    (make_recipe(1, "x", ["oat milk"], tags=["Lactose Free"]), "dairy-free", True),
    (make_recipe(1, "x", ["rice", "chicken"]), "gluten-free", True),  # keyword-only diets
    (make_recipe(1, "x", ["flour", "chicken"]), "gluten-free", False),
    (make_recipe(1, "x", ["walnuts"]), "nut-free", False),
    (make_recipe(1, "x", ["chicken"]), "pescatarian", True),  # unsupported: not filtered
])
def test_satisfies_diet(recipe, diet, expected):
    assert satisfies_diet(recipe, diet) is expected


def test_unsupported_diet():
    assert unsupported_diet(QueryConstraints(diet="pescatarian")) == "pescatarian"
    assert unsupported_diet(QueryConstraints(diet="Vegan")) is None


def test_ranking_text_leaves_out_exclusions_and_diet():
    c = QueryConstraints(free_text="pasta", include_ingredients=["basil"], exclude_ingredients=["mushrooms"],
                         diet="vegetarian")
    assert ranking_text(c, "original query") == "pasta basil"
    assert ranking_text(QueryConstraints(), "original query") == "original query"


# --- Search ---

def test_search_filters_then_ranks_by_cosine():
    recipes = [PASTA, MUSHROOM_PASTA, SLOW_PASTA, BACON_PASTA, make_recipe(5, "Tomato Soup", ["tomatoes"], minutes=15)]
    # 2-d unit vectors: x = "pasta-ness", y = "soup-ness"
    embeddings = np.array([[1, 0], [1, 0], [1, 0], [1, 0], [0, 1]], dtype=np.float32)
    encode = lambda texts: np.array([[0.8, 0.6]], dtype=np.float32)
    c = QueryConstraints(max_minutes=30, exclude_ingredients=["mushroom"], diet="vegetarian", free_text="pasta")
    results = search(c, "q", recipes, embeddings, encode, k=5)
    assert [r["recipe"]["id"] for r in results] == [1, 5]  # 2, 3, 4 filtered; pasta ranks above soup
    assert results[0]["score"] == pytest.approx(0.8)
    assert results[0]["missing_ingredients"] == []  # nothing requested


def test_search_returns_empty_when_nothing_passes():
    c = QueryConstraints(max_minutes=1)
    assert search(c, "q", [PASTA], np.ones((1, 2), dtype=np.float32), lambda t: np.ones((1, 2))) == []


def test_row_to_recipe_unescapes_html_in_name():
    assert row_to_recipe(csv_row(Name="Mac &amp; &quot;Cheese&quot;"))["name"] == 'Mac & "Cheese"'


# --- Included-ingredient boost ---

CHICKEN = make_recipe(10, "Greek Island Chicken", ["boneless skinless chicken breasts", "olive oil", "lemon"])
BRINE = make_recipe(11, "Mean Chef's Maple Brine", ["brown sugar", "maple syrup", "kosher salt"])
HUMMUS = make_recipe(12, "Hummus", ["canned chickpeas", "tahini", "lemon juice"])


def unit(*v):
    v = np.array(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_chicken_example_recipe_with_the_ingredient_wins():
    # The brine is more similar to "dinner" but has no chicken (the real top-5 result we saw).
    embeddings = np.stack([unit(0.5, 0.866), unit(0.9, 0.436)])
    encode = lambda texts: unit(1, 0)[None]
    c = QueryConstraints(free_text="dinner", include_ingredients=["chicken"])
    results = search(c, "chicken dinner with no dairy", [CHICKEN, BRINE], embeddings, encode)
    assert [r["recipe"]["id"] for r in results] == [10, 11]
    assert results[0]["similarity"] < results[1]["similarity"]
    assert [r["missing_ingredients"] for r in results] == [[], ["chicken"]]


def test_all_ingredients_beat_none_even_at_opposite_similarity():
    embeddings = np.stack([unit(-1, 0), unit(1, 0)])  # worst vs best possible similarity
    c = QueryConstraints(include_ingredients=["chicken"])
    results = search(c, "q", [CHICKEN, BRINE], embeddings, lambda t: unit(1, 0)[None])
    assert results[0]["recipe"]["id"] == 10


def test_include_matches_normalized_names():
    c = QueryConstraints(include_ingredients=["chickpeas"])
    results = search(c, "q", [BRINE, HUMMUS], np.stack([unit(1, 0), unit(0, 1)]), lambda t: unit(1, 0)[None])
    assert results[0]["recipe"]["id"] == 12 and results[0]["missing_ingredients"] == []


def test_partial_matches_are_returned_with_missing_ingredients():
    c = QueryConstraints(include_ingredients=["chicken", "lemon", "rice"])  # no recipe has all three
    embeddings = np.stack([unit(1, 0), unit(1, 0), unit(1, 0)])
    results = search(c, "q", [BRINE, HUMMUS, CHICKEN], embeddings, lambda t: unit(1, 0)[None])
    assert [(r["recipe"]["id"], r["missing_ingredients"]) for r in results] == [
        (10, ["rice"]),  # 2 of 3
        (12, ["chicken", "rice"]),  # "lemon juice" contains "lemon"
        (11, ["chicken", "lemon", "rice"]),
    ]


# --- Regex fallback parser ---

@pytest.mark.parametrize("query, diet, max_minutes, exclude, text", [
    ("vegetarian pasta under 30 minutes, no mushrooms", "vegetarian", 30, ["mushrooms"], "pasta"),
    ("quick vegan curry in under 45 minutes", "vegan", 45, [], "quick curry"),
    ("gluten-free chocolate dessert", "gluten-free", None, [], "chocolate dessert"),
    ("Dairy Free smoothie", "dairy-free", None, [], "smoothie"),
    ("nut-free cookies", "nut-free", None, [], "cookies"),
    ("low carb breakfast", "low-carb", None, [], "breakfast"),
    # time
    ("dinner in less than 1 hour", None, 60, [], "dinner"),
    ("breakfast in 15 min", None, 15, [], "breakfast"),
    ("30-minute meals", None, 30, [], "meals"),
    ("soup less than 25 mins", None, 25, [], "soup"),
    # exclusions
    ("salmon under 40 minutes without garlic", None, 40, ["garlic"], "salmon"),
    ("pasta, no mushrooms or olives, under 30 minutes", None, 30, ["mushrooms", "olives"], "pasta"),
    ("chili without beans and corn", None, None, ["beans", "corn"], "chili"),
    ("mushroom-free risotto", None, None, ["mushroom"], "risotto"),
    ("no mushroom pasta under 20 minutes", None, 20, ["mushroom"], "pasta"),  # dish word goes back to text
    ("no green peppers, please", None, None, ["green peppers"], ""),
    ("risotto without any cheese", None, None, ["cheese"], "risotto"),
    # negated diets
    ("chicken dinner with no dairy", "dairy-free", None, [], "chicken dinner"),
    ("no meat lasagna", "vegetarian", None, [], "lasagna"),
    # not exclusions
    ("no-bake cookies", None, None, [], "no-bake cookies"),
    ("no knead bread", None, None, [], "no knead bread"),
    ("vegan gluten-free cake", "vegan", None, [], "gluten-free cake"),  # first diet wins, second stays
    ("easy weeknight tacos", None, None, [], "easy weeknight tacos"),
])
def test_fallback_parse(query, diet, max_minutes, exclude, text):
    c = fallback_parse(query)
    assert (c.diet, c.max_minutes, c.exclude_ingredients, c.free_text) == (diet, max_minutes, exclude, text)


@pytest.mark.parametrize("query", [
    "vegetarian pasta under 30 minutes, no mushrooms", "salmon without garlic", "pasta, no mushrooms or olives",
    "chicken dinner with no dairy", "risotto without any cheese",
])
def test_fallback_negations_never_reach_the_embedding(query):
    text = ranking_text(fallback_parse(query), query)
    assert not re.search(r"\b(no|without)\b", text), text


def test_gluten_free_search_keeps_oats_and_notes_them():
    oat_bars = {"id": 1, "name": "Oat Bars", "minutes": 20, "tags": [], "ingredients": ["rolled oats", "oat flour", "honey"],
                "amount_hints": None, "steps": []}
    cake = {"id": 2, "name": "Cake", "minutes": 20, "tags": [], "ingredients": ["flour", "sugar"],
            "amount_hints": None, "steps": []}
    encode = lambda texts: np.array([[1.0, 0.0]], dtype=np.float32)
    embeddings = np.eye(2, dtype=np.float32)
    results = search(QueryConstraints(diet="gluten-free"), "bars", [oat_bars, cake], embeddings, encode)
    assert [r["recipe"]["id"] for r in results] == [1]
    assert results[0]["diet_notes"] == [{"ingredient": "rolled oats", "note": "use oats labeled gluten-free"},
                                        {"ingredient": "oat flour", "note": "use oats labeled gluten-free"}]
    assert search(QueryConstraints(), "bars", [oat_bars], embeddings[:1], encode)[0]["diet_notes"] == []


def test_groq_completer_sends_json_mode_request(monkeypatch):
    import groq
    from recipe_search import groq_completer
    sent = {}

    class FakeGroq:
        def __init__(self, api_key):
            sent["api_key"] = api_key
            self.chat = self
            self.completions = self

        def create(self, **kwargs):
            sent.update(kwargs)
            message = type("Message", (), {"content": '{"diet": "vegan"}'})()
            return type("Response", (), {"choices": [type("Choice", (), {"message": message})()]})()

    monkeypatch.setattr(groq, "Groq", FakeGroq)
    assert groq_completer(None) is None and groq_completer("") is None
    ask_llm = groq_completer("test-key", "some/model")
    assert parse_query("vegan soup", ask_llm) == (QueryConstraints(diet="vegan"), "llm")
    assert sent["api_key"] == "test-key" and sent["model"] == "some/model"
    assert sent["response_format"] == {"type": "json_object"} and sent["temperature"] == 0
    assert "vegan soup" in sent["messages"][0]["content"]
