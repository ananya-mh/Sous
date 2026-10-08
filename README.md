# Sous

## Meal planner: recipe search and grocery list (in progress)

Search real Food.com recipes from a plain-language request ("vegetarian pasta under 30 minutes, no
mushrooms"), pick one or more, optionally paste your own recipe, and get one combined grocery list
with substitutions for anything that breaks your constraints.

### Dataset and its limitations

The recipes come from `recipes.csv` in Kaggle's "Food.com Recipes and Reviews" (522,517 recipes).
The app loads a fixed-seed random sample of 5,000 (`RECIPES_LIMIT`) **from recipes with complete
ingredient lists only** (see below). Checked over the full file:

- **No units.** `RecipeIngredientQuantities` holds bare numbers ("1/4" next to "granulated sugar":
  cups? teaspoons?), so Food.com quantities can't be summed or converted.
- **Quantities and ingredients rarely line up.** Only 22% of recipes have quantity and ingredient
  lists of equal length. In the rest, Food.com dropped ingredient names it didn't recognize but
  kept their quantities, so pairing them by position puts amounts next to the wrong ingredient.
- **Missing quantities.** 35% of recipes have at least one `NA` quantity.
- **Ingredient names are dropped.** 74% of recipes list more quantities than ingredient names
  (median 2 names missing). Food.com seems to drop names it doesn't recognize. A "chocolate cake"
  can list only pudding mix, sour cream, eggs and butter, with the cake mix gone. For those
  recipes, diet and exclusion checks can pass because the offending ingredient isn't in the data,
  and their grocery-list items are incomplete.
**Restricting to complete recipes.** Because of the dropped names, the sample only includes recipes
with no more quantities than ingredient names: 136,741 recipes (26%). Does that skew the data?
Somewhat, mostly by category:

| | Complete (used) | Incomplete (skipped) |
|---|---|---|
| Total time, median (p25–p75) | 36 min (20–65) | 40 min (22–70) |
| Recipes ≤ 30 min | 44% | 40% |
| Ingredients, median | 8 | 9 (by quantity count) |
| Top categories | Dessert 12.3%, Vegetable 6.3%, Breakfast 5.0%, Lunch/Snacks 4.7% | Dessert 11.7%, Lunch/Snacks 6.8%, One Dish Meal 6.6%, Vegetable 4.9% |

Time and size barely move: complete recipes are slightly quicker and have one fewer ingredient.
The real skew is **meat**: Steak, Meat, Pork and Poultry are about half as common among complete
recipes (Meat 1.4% vs. 2.9%, Pork 1.4% vs. 2.8%). Breads and Sauces are more common (Breads 3.6% vs.
2.0%). Food.com probably drops cut names it doesn't recognize. Meat searches have a smaller pool,
and vegetarian searches lose little. Of the sample, 84% has amount hints (lists line up), up from 22%.

- **No vegetarian or gluten-free tags.** Diet tags exist for Vegan, Very Low Carbs, Lactose Free
  and Dairy Free Foods (plus Egg Free and Nuts), but not for vegetarian or gluten-free.

So the grocery list treats Food.com recipes **by name only**. Items are merged by normalized name
and shown as "flour (amount varies by recipe)". The raw amount is shown as a hint ("1/4, unit not
given") only for recipes whose lists line up. A **pasted recipe** has full lines ("2 cups chopped
onions"), so it goes through the BERT parser and real quantity merging, and the two are combined:
"2 cups onion + more for 1 other recipe (amount not given)".

The CSV was preferred over `recipes.parquet`: after parsing its R-style lists (`c("4", "1/4", NA)`),
it matches the parquet file on ingredients and keywords, without adding `pyarrow` as a dependency.

### Running it

Add to `.env` (see `.env.example`):

```bash
RECIPES_CSV=data/recipes.csv          # Kaggle "Food.com Recipes and Reviews" (irkaal)
RECIPES_LIMIT=5000
RECIPES_SUBSET_CACHE=data/recipes_subset.json
RECIPES_EMBEDDINGS_CACHE=data/recipe_embeddings.npz
GROQ_API_KEY=...                      # query parsing and substitutions (Groq-hosted open-weight model)
GROQ_MODEL=openai/gpt-oss-120b         # optional (default); gpt-oss-20b and qwen/qwen3.8-27b also work
BERT_MODEL_ID=fixed/model             # local model folder or Hub id; otherwise bert_recipe_model/
```

```bash
python scripts/api.py                 # first start: ~30 s to sample the CSV and embed; then cached
cd RecipeAI && npm install && npm run dev
```

Open the app, log in, and click **Meal Planner**. Without `RECIPES_CSV` the rest of the app works,
and the search endpoints return 503. Pasted recipes still work without it.

Endpoints (all need the usual `Authorization: Bearer <token>`):

- `POST /api/recipes/search` with `{"query": "..."}` returns the parsed `constraints`,
  `unsupported_diet`, and the top 5 `results` (`id`, `name`, `minutes`, `top_ingredients`,
  `missing_ingredients`, `diet_notes`).
- `GET /api/recipes/{id}` returns one recipe for the detail view: `ingredients` (each with its
  unitless `amount`, or null), `steps`, `minutes` and `tags`.
- `POST /api/recipes/grocery-list` with `{"recipe_ids": [...], "constraints": {...},
  "pasted_recipe": {"name": "...", "text": "one ingredient per line"}}` returns `items`,
  `unmerged` (couldn't be combined, with a reason), `substitutions`, and `parser` (`bert`, or
  `fallback` when the model isn't loaded). Items that the diet allows with care carry a
  `diet_note` (see "Oats and gluten-free" below).

Tests and evaluation:

```bash
python -m pytest                                   # unit and API tests (no network, DB or model needed)
python scripts/evaluate_search.py                  # 10 queries, parsed by the LLM (Groq)
python scripts/evaluate_search.py --no-llm         # same, with hand-written constraints
python scripts/evaluate_search.py --fallback-parser  # same, parsed by the regex fallback
python scripts/evaluate_search.py --fallback-parser --queries scripts/eval_queries_heldout.json
```

`--queries` takes a JSON file of `{"query": ..., "expected": {...}}` entries (format and an example
are in `scripts/eval_queries_heldout.json`). Results go to
`Results/search_eval_<mode>_<file>.json`, e.g. `search_eval_fallback_heldout.json`.

The evaluation checks each top-5 result against the query's hand-written constraints and reports
(1) how many satisfy every hard constraint, (2) how many contain all requested ingredients, and,
when a parser runs, (3) how many queries it parsed correctly (diet, time, exclusions). With a
parser, hard-constraint failures mean the query was parsed wrong. With `--no-llm`, (1) is 100%
by construction.

Built-in queries (10; written alongside the fallback parser's patterns):

| Run | Hard constraints | All requested ingredients | Parsed correctly |
|-----|-----|-----|-----|
| `--no-llm`, all recipes (before the restriction) | 50/50 | 22/30 (73%) | n/a |
| `--no-llm`, complete recipes only | 50/50 | 21/30 (70%) | n/a |
| `--fallback-parser`, complete recipes only | 50/50 | 21/30 (70%) | 10/10 |
| LLM (`openai/gpt-oss-120b`), complete recipes only | 50/50 | 21/30 (70%) | 7/10 |

Held-out queries (`scripts/eval_queries_heldout.json`, 15; written without looking at the
fallback's patterns). Complete recipes only:

| Run | Hard constraints | All requested ingredients | Parsed correctly |
|-----|-----|-----|-----|
| `--no-llm` | 75/75 (100%) | 25/30 (83%) | n/a |
| `--fallback-parser` | 36/75 (48%) | 15/30 (50%) | 4/15 |
| LLM (`openai/gpt-oss-120b`) | 75/75 (100%) | 25/30 (83%) | 12/15 |

On new phrasings the fallback parser falls apart (10/10 on the built-in queries, 4/15 here). It
misses "nothing over 40 min", "half an hour or less", "under an hour", "20 min snack", "skip the
onions", "hate mushrooms", "allergic to peanuts", and diet phrasings like "celiac" or "plant
based"; "no eggs and no nuts" leaves "no nuts" as an exclusion. Every hard-constraint violation in
that run comes from these misparses. It stays a fallback for when the LLM call fails.

The LLM's three misses don't change any results: twice it repeats the diet as an exclusion
("dairy" with dairy-free, "nuts" with nut-free; the same pattern as on the built-in queries), and
once it answers "keto" where the expected label is "low-carb" (the two filter identically, but the
check compares names). Its results match the hand-written constraints on every query.

The restriction barely changes these numbers, because this evaluation can't see dropped
ingredients: a recipe whose cake mix was dropped *looks* gluten-free to both the filter and the
check. What changes is the candidate pool. Gluten-free chocolate-dessert candidates went from 3,202
to 2,928, because recipes that passed only through missing names are gone. The included-ingredient
misses come from small pools: lentil-and-carrot soups are rare, and only 13 recipes are dairy-free
and take 10 minutes or less. The fallback's 10/10 is optimistic: its patterns were written
alongside these queries, so new phrasings will do worse; the held-out set measures that.

### How search ranks recipes

1. **Parse.** One LLM call (Groq, `openai/gpt-oss-120b` by default, in JSON mode) turns the
   request into `include_ingredients`, `exclude_ingredients`, `diet`, `max_minutes` and `free_text`. If the call or its JSON fails, a regex fallback parser
   takes over (`fallback_parse`), and the response says `"parser": "fallback"` so the UI can warn.
   It handles:
   - time: "under 30 minutes", "less than 1 hour", "in 15 min", "30-minute"
   - exclusions: "no X", "without X", "X-free"; lists ("no mushrooms or olives") are split, and dish
     words go back to the text ("no mushroom pasta" excludes mushroom and keeps "pasta")
   - diets: vegetarian, vegan, gluten-free, dairy-free, nut-free, low-carb, keto, plus negated
     diets ("no dairy" = dairy-free, "no meat" = vegetarian)

   Matched phrases are removed from `free_text`, so negations never reach the embedding. Limits:
   it doesn't extract included ingredients (they still rank through the text), only the first diet
   word is used, and phrasings outside these patterns are missed.
2. **Filter.** Hard filters: `max_minutes`, excluded ingredients (normalized whole-word match, so
   "mushrooms" excludes "cremini mushrooms"), and diet (see below).
3. **Rank.** Each recipe's name, tags and ingredients are embedded with all-MiniLM-L6-v2 and cached.
   The query side embeds only `free_text` and `include_ingredients`:
   - **Negations are left out.** Embeddings don't understand "no": embedding "no mushrooms" moves
     the query *toward* mushroom recipes. Exclusions are handled by the filter instead.
   - **Diet words are left out.** The diet is already a hard filter, and embedding "vegan" pulls
     toward anything vegan: "quick vegan curry" ranked vinaigrettes first until it was removed.
4. **Boost included ingredients.** The score is cosine similarity plus 2.01 x the fraction of
   requested ingredients the recipe contains (normalized match, so "chickpeas" matches "chickpea").
   Cosine similarity is between -1 and 1, so a recipe with every requested ingredient always
   outranks one with none. If no recipe has them all, the best partial matches are returned, and
   each result lists its `missing_ingredients`.

### Diet detection

Diet filtering combines Food.com tags with a simple ingredient keyword check
(`scripts/grocery_list.py`, `DIET_KEYWORDS`):

| Diet | How it's checked |
|------|------------------|
| vegan, low-carb / keto, dairy-free | Food.com tag (Vegan; Very Low Carbs; Lactose Free or Dairy Free Foods) **and** no off-limits ingredient keywords |
| vegetarian, gluten-free, nut-free | Ingredient keywords only (no tags exist) |
| anything else (e.g. pescatarian) | Not filtered; the API reports it as unsupported |

Known limitations of the keyword check:

- It only sees ingredient names, so hidden ingredients are missed (a "broth" that is chicken broth,
  store-bought sauces with wheat or dairy, Worcestershire sauce made with anchovies when written
  as just "sauce").
- Keywords match whole words, so "nut" doesn't catch "walnut" by itself (walnut is listed
  separately), and new or unusual names ("guanciale", "speck") aren't in the lists.
- Coconut isn't treated as a nut.
- "Safe" phrases ("peanut butter" isn't dairy, "almond flour" is gluten-free) are hand-listed per
  diet, so unlisted variants can be flagged wrongly.
- Oats are allowed but not verified: see below.
- Food.com tags are user-entered and sometimes wrong. That's why tag diets also run the keyword
  check (a Vegan-tagged recipe with honey is rejected).

**Oats and gluten-free.** Oats don't contain gluten, but most are processed alongside wheat. So
gluten-free searches don't exclude oat recipes (oats, oatmeal, oat flour, oat milk). Instead
they're flagged with `DIET_NOTES` in `grocery_list.py`: search results list them in `diet_notes`,
grocery-list items get `diet_note: "use oats labeled gluten-free"`, and the Meal Planner page
shows the note on the recipe card and next to the item. Oats aren't sent for substitution.

## Evaluation notes

The original BERT results in `Results/evaluation_report_table.json` (overall F1 0.926) overstate
how well the parser works, for three reasons:

- **Duplicate leakage.** `data/training_data.json` has 178,668 sentences but only 85,319 unique
  ones (ignoring case, which the uncased model can't see anyway). `train_BERT.py` splits at
  random *before* deduplicating, so most test sentences also appear in training. It also selects
  the best checkpoint on that same test split.
- **Label noise.** `preprocess.py` labels tokens by substring matching against the NYT `name` and
  `comment` columns, so `"a"` inside "banana" makes the token `a` a NAME (2,271 times), and `(`,
  `of`, `in` get entity labels too. It never emits `I-AMT`, `I-UNIT` or `I-DESC`, so "1 1/2" is
  two amounts and "finely chopped" is two descriptors. Test scores measure agreement with
  these rules, not with what a person would label.
- **Per-token, not per-entity.** The "per_entity_metrics" in the report are computed per token
  (AMT support 189,147 equals the number of `B-AMT` tokens), not per entity span.

### Retraining

`scripts/train_bert_colab.py` deduplicates first, splits 80/10/10 (train/val/test), selects the
checkpoint on val, and reports entity-level P/R/F1 per label on test. `--fix-labels` repairs the
noise above (filler and punctuation become `O`, and mixed numbers, ranges, "fluid ounces" and
consecutive descriptors become single spans). Both label modes use the same split.

```bash
pip install -r requirements-train.txt            # on Colab; torch is preinstalled there
python scripts/train_bert_colab.py               # -> Results/metrics_original.json, runs/original/model
python scripts/train_bert_colab.py --fix-labels  # -> Results/metrics_fixed.json,    runs/fixed/model
```

To use a trained model, copy it to `bert_recipe_model/`, or upload it to the Hugging Face Hub and
set `BERT_MODEL_ID=<user>/<repo>` (with `HF_TOKEN` if the repo is private). `api.py` tries
`BERT_MODEL_ID` first and falls back to the local folder.

### Why the parser uses `aggregation_strategy="first"`

BERT splits some words into subwords ("minced" becomes `min` + `##ced`). Training labels only
the first subword of each word and ignores the rest, so predictions for later subwords are
never trained. With `"simple"` aggregation, each subword keeps its own prediction, and words
split across fields: the trained model returned unit `"tables"` for "tablespoon" and descriptor
`"min ced"` for "minced". `"first"` labels the whole word with its first subword's prediction,
which matches how the model was trained. `api.py` and `inference.py` both use it, and
`tests/test_bert_aggregation.py` checks this.

One known gap: the training data keeps "1/2" as one token, but the pipeline splits it into
the words `1`, `/`, `2`. The fixed-labels model tags all three as AMT, so the amount still
reads as 1/2. The original-labels model tags `/` as O, giving "1 2". The grocery list falls back to
the leading amount in the original line when that happens.

### Gold set

Rule-labeled test sets can't show which label mode is actually better, so `gold/` holds
hand-corrected sentences: `gold/nyt.tsv` (50 sentences from the held-out test split, never in
train or val) and, later, `gold/foodcom.tsv` (50 Food.com lines). They start pre-labeled by
the rules: correct each label and set `# reviewed: yes`. Then:

```bash
python scripts/evaluate_gold.py --model runs/original/model --tag original
python scripts/evaluate_gold.py --model runs/fixed/model --tag fixed
```

This reports entity-level P/R/F1 per label for each subset, plus token-level F1. The token-level
score doesn't penalize span conventions, which matters for the original-labels model because
it was never trained to produce `I-` tags.

### Results

_Gold-set numbers are placeholders until the gold set is reviewed. Unreviewed, it is pre-labeled
with the fixed rules, which favors the fixed-labels model._

**Accuracy: entity-level F1 on the reviewed gold set** (`Results/gold_metrics_*.json`):

| Label | Original labels (NYT) | Fixed labels (NYT) | Original labels (Food.com) | Fixed labels (Food.com) |
|-------|-----|-----|-----|-----|
| AMT   | TBD | TBD | TBD | TBD |
| UNIT  | TBD | TBD | TBD | TBD |
| NAME  | TBD | TBD | TBD | TBD |
| DESC  | TBD | TBD | TBD | TBD |
| Micro avg | TBD | TBD | TBD | TBD |

**Agreement with rule-made labels (not accuracy).** Entity-level F1 on the deduplicated test
split, from `Results/metrics_*.json`. The test labels come from the same rules as the training
labels, so these numbers measure how well each model reproduces its rules. Accuracy comes from
the reviewed gold set above.

| Label | Original labels | Fixed labels |
|-------|-----|-----|
| AMT   | 0.9995 | 0.9993 |
| UNIT  | 0.9990 | 0.9990 |
| NAME  | 0.7689 | 0.7776 |
| DESC  | 0.8546 | 0.8189 |
| Micro avg | 0.8880 | 0.8829 |

For comparison, `train_BERT.py` reported a micro F1 of 0.926 on its leaky split (also agreement with
rule labels). The same
recipe scores 0.888 once duplicates are removed and the checkpoint is chosen on val. AMT and
UNIT are near 1.0 because the rules tag every number and every listed unit, so the model only
has to learn those rules. Each run is scored against its own label version, so the two
columns can't be compared directly. Scoring both models against the same labels, each agrees
best with its own version (vs. original labels: 0.888 original, 0.670 fixed; vs. fixed labels: 0.657
original, 0.883 fixed). On rule labels, both models mostly reproduce the rules they were trained
on. Token-level scores, which ignore span conventions, are close (NAME 0.87–0.88 for both). The
practical difference is spans: the fixed model returns "2 to 3" and "2 1/2" as one amount. Use
the reviewed gold set to pick a winner.
