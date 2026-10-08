"""Recipe search over a Food.com subset: CSV loading, embeddings, query parsing, filtering, ranking.

Data: recipes.csv from Kaggle "Food.com Recipes and Reviews" (irkaal). List columns are R vectors
written as text, e.g. c("4", "1/4", NA). Quantities have no units, and the quantity and ingredient
lists line up only when they have the same length, so quantities are kept as display hints only.
"""
import csv
import hashlib
import html
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator

from grocery_list import canonical_diet, conflicts_with_diet, diet_note, ingredient_matches

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SUBSET_FORMAT = 3  # bump when row_to_recipe changes, so cached subsets are rebuilt

# --- CSV parsing ---

_R_TOKEN = re.compile(r'"((?:[^"\\]|\\.)*)"|\bNA\b')
_ISO_DURATION = re.compile(r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$")


def parse_r_vector(text):
    """'c("a", NA)' -> ['a', None]. One-element vectors are written as a quoted scalar ('"a"')."""
    if text in ("", "character(0)"):
        return []
    if text == "NA":
        return [None]
    if text.startswith("c(") and text.endswith(")"):
        return [_unescape(m.group(1)) if m.group(1) is not None else None for m in _R_TOKEN.finditer(text[2:-1])]
    if len(text) >= 2 and text[0] == text[-1] == '"':
        return [_unescape(text[1:-1])]
    return [text]


def _unescape(s):
    s = s.replace('\\"', '"').replace("\\r\\n", "\n").replace("\\n", "\n")
    return html.unescape(s)


def iso_minutes(duration):
    """'PT1H30M' -> 90. Returns None if it doesn't parse."""
    m = _ISO_DURATION.match(duration or "")
    if not m or not any(m.groups()):
        return None
    days, hours, minutes, seconds = (int(g) if g else 0 for g in m.groups())
    return days * 1440 + hours * 60 + minutes + (1 if seconds else 0)


def row_to_recipe(row):
    """One CSV row -> recipe dict, or None if it has no ingredients, no usable time, or an
    incomplete ingredient list.

    More quantities than ingredient names means Food.com dropped names (74% of recipes). Those
    are skipped: a dropped name could be the meat, gluten or excluded ingredient that the filters
    need to see, and their grocery lists would be missing items.
    """
    parts = [p for p in parse_r_vector(row["RecipeIngredientParts"]) if p]
    minutes = iso_minutes(row["TotalTime"])
    if not parts or not minutes:
        return None
    quantities = parse_r_vector(row["RecipeIngredientQuantities"])
    raw_parts = parse_r_vector(row["RecipeIngredientParts"])
    if len(quantities) > len(raw_parts):
        return None
    aligned = len(quantities) == len(raw_parts) and all(raw_parts)  # same length, nothing dropped
    tags = [t for t in parse_r_vector(row["Keywords"]) if t]
    if row.get("RecipeCategory") and row["RecipeCategory"] != "NA":
        tags.append(row["RecipeCategory"])
    return {
        "id": int(float(row["RecipeId"])),
        "name": html.unescape(row["Name"]),  # names contain "&quot;", "&amp;"
        "minutes": minutes,
        "tags": tags,
        "ingredients": parts,
        # Unitless amounts, only when the lists line up; "1⁄4" (fraction slash) -> "1/4".
        "amount_hints": [q.replace("⁄", "/") if q else None for q in quantities] if aligned else None,
        "steps": [s for s in parse_r_vector(row["RecipeInstructions"]) if s],
    }


def _sample_csv(csv_path, limit, seed):
    """Fixed-seed reservoir sample of usable recipes, so the subset is spread over the whole file."""
    csv.field_size_limit(sys.maxsize)
    rng, sample, seen = random.Random(seed), [], 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            recipe = row_to_recipe(row)
            if recipe is None:
                continue
            seen += 1
            if len(sample) < limit:
                sample.append(recipe)
            else:
                j = rng.randrange(seen)
                if j < limit:
                    sample[j] = recipe
    sample.sort(key=lambda r: r["id"])
    return sample


def load_recipes(csv_path, limit=5000, seed=42, cache_path=None):
    """Load a `limit`-recipe subset, cached as JSON keyed by the CSV's size/mtime, limit and seed."""
    stat = os.stat(csv_path)
    key = {"format": SUBSET_FORMAT, "csv_size": stat.st_size, "csv_mtime": int(stat.st_mtime),
           "limit": limit, "seed": seed}
    if cache_path and Path(cache_path).exists():
        cached = json.loads(Path(cache_path).read_text())
        if cached.get("key") == key:
            return cached["recipes"]
    recipes = _sample_csv(csv_path, limit, seed)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps({"key": key, "recipes": recipes}))
    return recipes


# --- Embeddings ---

def recipe_text(recipe):
    return f"{recipe['name']}. Tags: {', '.join(recipe['tags'])}. Ingredients: {', '.join(recipe['ingredients'])}"


def sentence_encoder(model_name=EMBEDDING_MODEL):
    """encode(texts) -> unit-length float32 vectors. Imported lazily: it loads torch."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return lambda texts: model.encode(list(texts), batch_size=64, normalize_embeddings=True,
                                      convert_to_numpy=True).astype(np.float32)


def load_or_build_embeddings(recipes, cache_path, encode, model_name=EMBEDDING_MODEL):
    """Embeddings for `recipes`, cached in an .npz keyed by the exact embedded texts and model name."""
    texts = [recipe_text(r) for r in recipes]
    digest = hashlib.sha256("\n".join([model_name, *texts]).encode()).hexdigest()
    if cache_path and Path(cache_path).exists():
        cached = np.load(cache_path)
        if str(cached["digest"]) == digest:
            return cached["embeddings"]
    embeddings = encode(texts)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, embeddings=embeddings, digest=np.array(digest))
    return embeddings


# --- Query parsing ---

class QueryConstraints(BaseModel):
    include_ingredients: list[str] = Field(default_factory=list)
    exclude_ingredients: list[str] = Field(default_factory=list)
    diet: Optional[str] = None
    max_minutes: Optional[int] = None
    free_text: str = ""

    @field_validator("max_minutes")
    @classmethod
    def positive_minutes(cls, v):
        return v if v is None or v > 0 else None

    @field_validator("include_ingredients", "exclude_ingredients")
    @classmethod
    def strip_empty(cls, v):
        return [s.strip() for s in v if s and s.strip()]


QUERY_PROMPT = """
Extract recipe search constraints from the user's request.

Request: {query}

Return ONLY valid JSON with exactly these keys:
{{
    "include_ingredients": ["ingredients the user wants"],
    "exclude_ingredients": ["ingredients the user wants to avoid, e.g. 'no mushrooms' -> 'mushrooms'"],
    "diet": "one of vegetarian, vegan, gluten-free, dairy-free, nut-free, low-carb, keto, or null",
    "max_minutes": "total time limit in minutes as an integer, or null",
    "free_text": "the rest of the request: dish type, cuisine, style (e.g. 'pasta')"
}}
"""


DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


def groq_completer(api_key, model=DEFAULT_GROQ_MODEL):
    """ask_llm(prompt) -> reply text, using Groq's chat API in JSON mode. None without a key."""
    if not api_key:
        return None
    from groq import Groq
    client = Groq(api_key=api_key)

    def ask_llm(prompt):
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}, temperature=0)
        return response.choices[0].message.content

    return ask_llm


def parse_query(query, ask_llm):
    """One LLM call -> (QueryConstraints, "llm"). If there's no LLM, or the call or its JSON fails,
    falls back to fallback_parse() -> (QueryConstraints, "fallback")."""
    try:
        if ask_llm is None:
            raise RuntimeError("GROQ_API_KEY is not set")
        clean_json = ask_llm(QUERY_PROMPT.format(query=query)).replace("```json", "").replace("```", "").strip()
        return QueryConstraints.model_validate_json(clean_json), "llm"
    except Exception as e:
        print(f"⚠️ Query parsing failed, using the fallback parser: {e}")
        return fallback_parse(query), "fallback"


# --- Regex fallback parser ---

_DIET_WORDS = re.compile(r"\b(vegetarian|vegan|gluten[- ]free|dairy[- ]free|nut[- ]free|low[- ]carb|keto)\b", re.I)
_TIME = re.compile(
    r"\b(?:(?:in\s+)?under|less\s+than|in|within|at\s+most|no\s+more\s+than|max(?:imum)?)\s+"
    r"(\d+)\s*(minutes?|mins?|m|hours?|hrs?|h)\b"
    r"|\b(\d+)[- ]minutes?\b", re.I)
# "no"/"without" + up to the next comma, period, or constraint keyword. "and"/"or" inside it split a list.
_NEGATION = re.compile(r"\b(?:with\s+)?(?:no|without)\s+([a-z][a-z' ]*?)(?=\s*(?:[,.;!?]|\b(?:under|less|in|within|with|for|that|please)\b|$))", re.I)
_X_FREE = re.compile(r"\b([a-z]+)-free\b", re.I)
_NOT_EXCLUSIONS = {"bake", "cook", "fuss", "knead", "churn", "boil", "fail"}  # "no-bake", "no knead bread"
# "no dairy" is a diet, not an ingredient exclusion.
# "no mushroom pasta": the negation covers "mushroom"; the dish word goes back to the text.
_DISH_WORDS = {"pasta", "lasagna", "soup", "salad", "curry", "pizza", "bread", "cake", "cookies", "cookie", "chili",
               "stew", "tacos", "taco", "casserole", "burger", "burgers", "sandwich", "sandwiches", "dinner", "lunch",
               "breakfast", "dessert", "meal", "meals", "dish", "recipe", "recipes", "risotto", "stir-fry", "smoothie"}
_NEGATED_DIETS = {"dairy": "dairy-free", "gluten": "gluten-free", "nuts": "nut-free", "nut": "nut-free",
                  "meat": "vegetarian", "carbs": "low-carb"}


def fallback_parse(query):
    """Pattern-based constraints for when LLM parsing fails.

    Covers time limits ("under 30 minutes", "less than 1 hour", "in 15 min", "30-minute"),
    exclusions ("no X", "without X", "X-free") and diet words (vegetarian, vegan, gluten-free,
    dairy-free, nut-free, low-carb, keto). Matched phrases are removed from free_text, so
    negations never reach the embedding. The first diet found wins; any others stay in the text.
    Included ingredients aren't extracted.
    """
    text, diet, max_minutes, exclude = query, None, None, []

    m = _DIET_WORDS.search(text)
    if m:
        diet = canonical_diet(m.group(1).lower().replace(" ", "-"))
        text = text[:m.start()] + " " + text[m.end():]

    m = _TIME.search(text)
    if m:
        number, unit = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), "minutes")
        max_minutes = int(number) * (60 if unit.lower().startswith("h") else 1)
        text = text[:m.start()] + " " + text[m.end():]

    def take_negation(match):
        nonlocal diet
        terms = [t.strip() for t in re.split(r"\s+(?:and|or)\s+|,", match.group(1)) if t.strip()]
        if not terms or terms[0].split()[0].lower() in _NOT_EXCLUSIONS:
            return match.group(0)
        kept = []  # dish words that were caught in the phrase go back to the text
        for term in terms:
            words = re.sub(r"^(?:any|the|a|an|some)\s+", "", term, flags=re.I).split()  # "without any cheese"
            cut = next((i for i, w in enumerate(words) if w.lower() in _DISH_WORDS), len(words))
            term, kept = " ".join(words[:cut]), kept + words[cut:]
            if not term:
                continue
            if term.lower() in _NEGATED_DIETS:
                diet = diet or _NEGATED_DIETS[term.lower()]
            else:
                exclude.append(term)
        return " " + " ".join(kept) + " "
    text = _NEGATION.sub(take_negation, text)

    def take_x_free(match):
        if canonical_diet(f"{match.group(1)}-free"):  # a second diet word ("vegan gluten-free"): leave it
            return match.group(0)
        exclude.append(match.group(1))
        return " "
    text = _X_FREE.sub(take_x_free, text)

    # Tidy what's left: collapse whitespace and drop dangling punctuation and connectors.
    text = re.sub(r"\s+([,.;!?])", r"\1", re.sub(r"\s+", " ", text)).strip(" ,.;!?")
    text = re.sub(r"(?:\s*,)+", ",", text)
    text = re.sub(r"\b(?:please|thanks|thank you)\b", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(" ,.;!?")
    text = re.sub(r"^(?:and|or|with|in)\b\s*|\s*\b(?:and|or|with|in)$", "", text, flags=re.I).strip(" ,.;!?")
    return QueryConstraints(diet=diet, max_minutes=max_minutes, exclude_ingredients=exclude, free_text=text)


# --- Filtering and ranking ---

# Food.com has tags for these diets; vegetarian, gluten-free and nut-free have none.
DIET_TAGS = {
    "vegan": {"vegan"},
    "low-carb": {"very low carbs"},
    "keto": {"very low carbs"},
    "dairy-free": {"lactose free", "dairy free foods"},
}


def satisfies_diet(recipe, diet):
    """Tag check where Food.com has a tag, plus the ingredient keyword check for every diet."""
    diet = canonical_diet(diet)
    if diet is None:
        return True
    if any(conflicts_with_diet(i, diet) for i in recipe["ingredients"]):
        return False
    if diet in DIET_TAGS:
        return bool(DIET_TAGS[diet] & {t.lower() for t in recipe["tags"]})
    return True


def violations(recipe, constraints):
    """Names of the hard constraints this recipe breaks (empty list = passes)."""
    broken = []
    if constraints.max_minutes and recipe["minutes"] > constraints.max_minutes:
        broken.append("max_minutes")
    if any(ingredient_matches(i, ex) for ex in constraints.exclude_ingredients for i in recipe["ingredients"]):
        broken.append("exclude_ingredients")
    if not satisfies_diet(recipe, constraints.diet):
        broken.append("diet")
    return broken


def ranking_text(constraints, query):
    """What to embed for ranking. Leaves out exclusions (embedding 'no mushrooms' pulls toward mushrooms)
    and the diet, which is already a hard filter ('vegan' pulls toward any vegan recipe, e.g. dressings)."""
    parts = [constraints.free_text, *constraints.include_ingredients]
    text = " ".join(p for p in parts if p).strip()
    return text or query


# Cosine similarity of unit vectors is in [-1, 1], so a boost just over 2 guarantees a recipe with
# every requested ingredient outranks one with none, whatever their similarities.
INCLUDE_BOOST = 2.01


def recipe_contains(recipe, term):
    """Normalized whole-word match, so "chickpeas" matches "canned chickpea"."""
    return any(ingredient_matches(i, term) for i in recipe["ingredients"])


def recipe_diet_notes(recipe, diet):
    """[{ingredient, note}] for ingredients the diet allows with care, e.g. oats in a gluten-free search."""
    notes = ((i, diet_note(i, diet)) for i in recipe["ingredients"])
    return [{"ingredient": i, "note": n} for i, n in notes if n]


def search(constraints, query, recipes, embeddings, encode, k=5):
    """Hard-filter, then rank by cosine similarity plus INCLUDE_BOOST x the fraction of
    include_ingredients each recipe contains. Partial matches are still returned when no recipe
    has them all; each result lists its missing_ingredients, and diet_notes (oats when gluten-free).
    """
    candidates = [i for i, r in enumerate(recipes) if not violations(r, constraints)]
    if not candidates:
        return []
    query_vec = encode([ranking_text(constraints, query)])[0]
    similarity = embeddings[candidates] @ query_vec
    wanted = constraints.include_ingredients
    missing = [[w for w in wanted if not recipe_contains(recipes[i], w)] for i in candidates]
    fraction = np.array([1 - len(m) / len(wanted) if wanted else 0.0 for m in missing])
    scores = similarity + INCLUDE_BOOST * fraction
    order = np.argsort(-scores, kind="stable")[:k]
    return [{"recipe": recipes[candidates[i]], "score": float(scores[i]), "similarity": float(similarity[i]),
             "missing_ingredients": missing[i],
             "diet_notes": recipe_diet_notes(recipes[candidates[i]], constraints.diet)} for i in order]


def unsupported_diet(constraints):
    """The diet text if we can't filter on it (e.g. 'pescatarian'), so the API can say so."""
    return constraints.diet if constraints.diet and canonical_diet(constraints.diet) is None else None
