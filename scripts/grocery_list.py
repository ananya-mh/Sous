"""Combined grocery list: normalize ingredient names, merge quantities, flag substitutions.

Parsing is injected as `parse_fn(text) -> {"amount", "unit", "item", "descriptor"}`, the same
contract as `parse_recipe_bert` in api.py. `simple_parse` is a regex stand-in with that
signature for when the BERT weights aren't available.
"""
import re
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import pint

ureg = pint.UnitRegistry()

# --- Name normalization ---

PREP_WORDS = {
    "chopped", "diced", "minced", "sliced", "grated", "shredded", "crushed", "cubed",
    "julienned", "halved", "quartered", "peeled", "seeded", "cored", "trimmed", "rinsed",
    "drained", "softened", "melted", "beaten", "sifted", "toasted", "divided", "packed",
    "fresh", "freshly", "finely", "coarsely", "roughly", "thinly", "thickly", "lightly",
    "large", "medium", "small", "whole", "optional", "about", "a", "an",
}

IRREGULAR_PLURALS = {
    "leaves": "leaf", "halves": "half", "loaves": "loaf", "cookies": "cookie",
    "brownies": "brownie", "calves": "calf", "knives": "knife",
}
# Words that end in "s" but are already singular (or uncountable).
NO_SINGULAR = {
    "molasses", "asparagus", "hummus", "couscous", "swiss", "grits", "greens", "citrus",
    "bitters", "brussels", "watercress",
}
# Different names for the same grocery item, applied to the fully normalized name.
SYNONYMS = {
    "scallion": "green onion", "spring onion": "green onion",
    "garbanzo bean": "chickpea", "garbanzo": "chickpea",
    "cilantro leaf": "cilantro", "coriander leaf": "cilantro",
    "confectioners sugar": "powdered sugar", "confectioner's sugar": "powdered sugar",
    "icing sugar": "powdered sugar",
    "courgette": "zucchini", "aubergine": "eggplant", "capsicum": "bell pepper",
    "all-purpose flour": "flour", "all purpose flour": "flour", "plain flour": "flour",
    "whipping cream": "heavy cream", "heavy whipping cream": "heavy cream",
    "bicarbonate of soda": "baking soda", "corn starch": "cornstarch", "cornflour": "cornstarch",
    "extra virgin olive oil": "olive oil", "extra-virgin olive oil": "olive oil",
}


def singularize(word):
    """Singularize one English word with a few rules; good enough for ingredient nouns."""
    if word in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[word]
    if word in NO_SINGULAR or len(word) <= 3:
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("oes"):
        return word[:-2]
    if re.search(r"(ch|sh|ss|x|z)es$", word):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def pluralize(word):
    if word.endswith("y") and word[-2:-1] not in "aeiou":
        return word[:-1] + "ies"
    if word.endswith(("ch", "sh", "s", "x", "z", "o")):
        return word + "es"
    return word + "s"


_SERVING_PHRASES = re.compile(
    r"\b(to taste|as needed|if needed|for garnish|for serving|for frying|for greasing|or more|or less)\b")


def normalize_name(name):
    """'2 Large Tomatoes, diced (about 1 lb)' style names -> 'tomato'."""
    text = name.lower()
    text = re.sub(r"\([^)]*\)", " ", text)  # drop parentheticals
    text = text.split(",")[0]  # "onion, finely chopped" -> "onion"
    text = _SERVING_PHRASES.sub(" ", text)  # "salt to taste" -> "salt"
    words = re.findall(r"[a-z]+(?:[-'][a-z]+)*", text)
    words = [w for w in words if w not in PREP_WORDS]
    if not words:
        return ""
    words[-1] = singularize(words[-1])
    name = " ".join(words)
    return SYNONYMS.get(name, name)


# --- Quantity parsing ---

UNICODE_FRACTIONS = {
    "½": "1/2", "⅓": "1/3", "⅔": "2/3", "¼": "1/4", "¾": "3/4", "⅕": "1/5",
    "⅛": "1/8", "⅜": "3/8", "⅝": "5/8", "⅞": "7/8",
}
WORD_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "dozen": 12,
}


def _parse_number(token):
    if token in WORD_NUMBERS:
        return float(WORD_NUMBERS[token])
    try:
        return float(Fraction(token))
    except (ValueError, ZeroDivisionError):
        return None


def _parse_single(text):
    """'1 1/2' -> 1.5. Returns None for anything ambiguous, like '1 14' from '1 (14 oz) can'."""
    tokens = text.split()
    if not tokens:
        return None
    values = [_parse_number(t) for t in tokens]
    if any(v is None for v in values):
        return None
    if len(values) == 1:
        return values[0]
    if len(values) == 2 and values[0] == int(values[0]) and 0 < values[1] < 1:
        return values[0] + values[1]  # mixed number
    return None


def parse_quantity(text):
    """Parse an amount string into a float. Ranges ('2-3', '2 to 3') return the upper bound."""
    if not text:
        return None
    s = text.lower().strip()
    for char, frac in UNICODE_FRACTIONS.items():
        s = s.replace(char, f" {frac}")
    s = re.sub(r"\s*/\s*", "/", s)  # BERT detokenizes '1/4' as '1 / 4'
    s = s.replace("–", "-").replace("—", "-")
    parts = [p for p in re.split(r"\s*-\s*|\s+(?:to|or)\s+", s) if p.strip()]
    if len(parts) == 2:
        low, high = _parse_single(parts[0]), _parse_single(parts[1])
        if low is not None and high is not None:
            # '1-1/2' is a mixed number, not the range 1 to 0.5
            return low + high if high < 1 <= low else max(low, high)
        return None
    if len(parts) == 1:
        return _parse_single(parts[0])
    return None


# --- Units ---

UNIT_ALIASES = {
    "cup": ["cup", "cups", "c", "c."],
    "tablespoon": ["tablespoon", "tablespoons", "tbsp", "tbsp.", "tbs", "tbs.", "tbl", "tbl."],
    "teaspoon": ["teaspoon", "teaspoons", "tsp", "tsp.", "t."],
    "fluid_ounce": ["fluid ounce", "fluid ounces", "fl oz", "fl. oz.", "fl oz.", "floz"],
    "pint": ["pint", "pints", "pt", "pt."],
    "quart": ["quart", "quarts", "qt", "qt."],
    "gallon": ["gallon", "gallons", "gal"],
    "milliliter": ["milliliter", "milliliters", "millilitre", "millilitres", "ml", "ml."],
    "liter": ["liter", "liters", "litre", "litres", "l"],
    "ounce": ["ounce", "ounces", "oz", "oz."],
    "pound": ["pound", "pounds", "lb", "lbs", "lb.", "lbs."],
    "gram": ["gram", "grams", "g", "g.", "gr"],
    "kilogram": ["kilogram", "kilograms", "kg", "kg."],
}
UNIT_LOOKUP = {alias: canon for canon, aliases in UNIT_ALIASES.items() for alias in aliases}
UNIT_DISPLAY = {
    "cup": "cup", "tablespoon": "tbsp", "teaspoon": "tsp", "fluid_ounce": "fl oz",
    "pint": "pint", "quart": "quart", "gallon": "gallon", "milliliter": "ml", "liter": "l",
    "ounce": "oz", "pound": "lb", "gram": "g", "kilogram": "kg",
}
# Units that read better pluralized ("2 cups"); abbreviations never are.
PLURALIZE = {"cup", "pint", "quart", "gallon"}
COUNT_UNITS = {
    "clove", "can", "package", "pkg", "slice", "pinch", "dash", "stick", "bunch", "head",
    "sprig", "piece", "jar", "bottle", "box", "bag", "container", "envelope", "packet",
    "stalk", "fillet", "ear", "drop", "handful", "sheet", "cube", "loaf",
}


def canonical_unit(unit):
    """Return (unit, is_measurable). Measurable units convert with pint; others match by name."""
    raw = unit.strip()
    if raw in ("T", "T."):  # capital T is tablespoon in recipe shorthand, lowercase t is teaspoon
        return "tablespoon", True
    if raw in ("t", "t."):
        return "teaspoon", True
    u = re.sub(r"\s+", " ", raw.lower())
    if u in UNIT_LOOKUP:
        return UNIT_LOOKUP[u], True
    return " ".join(singularize(w) for w in u.split()), False


def format_amount(value):
    """1.5 -> '1 1/2', 0.333 -> '1/3', 2.37 -> '2.37'."""
    whole = int(value)
    frac = value - whole
    for denom in (2, 3, 4, 8):
        num = round(frac * denom)
        if abs(frac - num / denom) < 0.02:
            if num == 0:
                if whole == 0:
                    continue  # tiny amount: don't round to "0"
                return str(whole)
            if num == denom:
                return str(whole + 1)
            f = Fraction(num, denom)
            return f"{whole} {f}" if whole else str(f)
    return f"{value:.2f}".rstrip("0").rstrip(".")


def format_quantity(value, unit, measurable):
    amount = format_amount(value)
    if not unit:
        return amount
    if measurable:
        label = UNIT_DISPLAY[unit]
        if unit in PLURALIZE and value > 1:
            label += "s"
    else:
        label = unit if value <= 1 else pluralize(unit)
    return f"{amount} {label}"


# --- Stand-in parser ---

_AMOUNT_RE = re.compile(
    r"^\s*((?:[\d½⅓⅔¼¾⅕⅛⅜⅝⅞]+(?:\s*[./]\s*\d+)?\s*(?:(?:-|to)\s*)?)+)", re.IGNORECASE
)


def simple_parse(text):
    """Regex stand-in for parse_recipe_bert, with the same return shape."""
    parsed = {"amount": "", "unit": "", "item": "", "descriptor": ""}
    rest = text.strip()
    m = _AMOUNT_RE.match(rest)
    if m:
        parsed["amount"] = m.group(1).strip().rstrip("-").strip()
        rest = rest[m.end():]
    rest = re.sub(r"\([^)]*\)", " ", rest).strip()

    words = rest.split()
    for n in (2, 1):  # try two-word units first ("fluid ounces")
        candidate = " ".join(words[:n])
        unit, measurable = canonical_unit(candidate) if candidate else ("", False)
        if candidate and (measurable or unit in COUNT_UNITS):
            parsed["unit"] = candidate
            words = words[n:]
            break
    rest = " ".join(words)

    name_part, _, comment = rest.partition(",")
    name_words, desc_words = [], []
    for w in name_part.split():
        (desc_words if w.lower() in PREP_WORDS - {"a", "an"} else name_words).append(w)
    if comment.strip():
        desc_words.append(comment.strip())
    parsed["item"] = " ".join(name_words).strip()
    parsed["descriptor"] = " ".join(desc_words).strip()
    return parsed


# --- Merging ---

def _entry(recipe, source, name, quantity=None, unit="", measurable=False, name_only=False, hint=None):
    return {"recipe": recipe, "source": source, "name": name, "quantity": quantity,
            "unit": unit, "measurable": measurable, "name_only": name_only, "hint": hint}


def _split_count_unit(name):
    """'garlic clove' -> ('garlic', 'clove'); names without a trailing count unit are unchanged."""
    head, _, last = name.rpartition(" ")
    if head and last in COUNT_UNITS:
        return SYNONYMS.get(head, head), last
    return name, ""


def _parse_lines(recipe, parse_fn):
    """Turn one recipe into entries. Recipes without usable lines fall back to name-only."""
    entries, unparsed = [], []
    if recipe.get("lines") is None:
        names = recipe.get("ingredient_names", [])
        hints = recipe.get("amount_hints") or [None] * len(names)
        for raw, hint in zip(names, hints):
            name, _ = _split_count_unit(normalize_name(raw))  # "garlic clove" -> "garlic"
            if name:
                entries.append(_entry(recipe["name"], raw, name, name_only=True, hint=hint))
        return entries, unparsed

    for line in recipe["lines"]:
        parsed = parse_fn(line)
        name = normalize_name(parsed.get("item", ""))
        if not name:
            unparsed.append({"name": line, "display": line, "recipes": [recipe["name"]],
                             "sources": [line], "reason": "couldn't identify the ingredient"})
            continue
        quantity = parse_quantity(parsed.get("amount", ""))
        if quantity is None:
            # BERT's pipeline splits "1/2" into "1", "/", "2" and tags "/" as O, so the amount
            # comes back as "1 2". Fall back to the leading amount in the original line.
            m = _AMOUNT_RE.match(line)
            if m:
                quantity = parse_quantity(m.group(1).strip().rstrip("-").strip())
        if quantity is None and parsed.get("amount"):
            unparsed.append({"name": name, "display": line, "recipes": [recipe["name"]],
                             "sources": [line], "reason": "couldn't read the quantity"})
            continue
        unit, measurable = canonical_unit(parsed.get("unit", "")) if parsed.get("unit") else ("", False)
        if not unit:
            name, unit = _split_count_unit(name)  # "2 garlic cloves" -> 2 clove garlic
        entries.append(_entry(recipe["name"], line, name, quantity, unit, measurable))
    return entries, unparsed


def _group_key(entry):
    """Entries with the same key can be summed: pint dimensionality, or the literal count unit."""
    if entry["measurable"]:
        return ("dim", str(ureg.Quantity(1, entry["unit"]).dimensionality))
    return ("count", entry["unit"])


METRIC_UNITS = {"gram", "kilogram", "milliliter", "liter"}
STANDARD_CUPS = (1 / 4, 1 / 3, 1 / 2, 2 / 3, 3 / 4)  # measuring-cup sizes
CUP_TOLERANCE = 0.01  # cups, about 1/2 tsp
_EPS = 1e-6  # float slack, so 3 tsp counts as a full tbsp


def display_unit(total, units):
    """Pick the unit to show a summed quantity in (a pint Quantity, plus the units that went in).

    All-metric groups stay metric (g/kg, ml/l, switching at 1000). Otherwise volume shows in
    cups at 1 cup or more, or when it's a measuring-cup size (1/4, 1/3, 1/2, 2/3, 3/4);
    else tbsp if at least 1 tbsp, else tsp. Weight shows in lb at 1 lb or more, else oz.
    """
    metric = set(units) <= METRIC_UNITS
    if total.check("[volume]"):
        if metric:
            return "liter" if total.to("milliliter").magnitude >= 1000 - _EPS else "milliliter"
        cups = total.to("cup").magnitude
        if cups >= 1 - _EPS or any(abs(cups - size) <= CUP_TOLERANCE for size in STANDARD_CUPS):
            return "cup"
        return "tablespoon" if total.to("tablespoon").magnitude >= 1 - _EPS else "teaspoon"
    if total.check("[mass]"):
        if metric:
            return "kilogram" if total.to("gram").magnitude >= 1000 - _EPS else "gram"
        return "pound" if total.to("pound").magnitude >= 1 - _EPS else "ounce"
    return units[0]


def _merge_group(name, entries):
    recipes = sorted({e["recipe"] for e in entries})
    sources = [e["source"] for e in entries]
    first = entries[0]
    if first["measurable"]:
        summed = ureg.Quantity(first["quantity"], first["unit"])
        for e in entries[1:]:
            summed = summed + ureg.Quantity(e["quantity"], e["unit"])  # pint converts to summed's unit
        unit = display_unit(summed, [e["unit"] for e in entries])
        total = summed.to(unit).magnitude
    else:
        total = sum(e["quantity"] for e in entries)
        unit = first["unit"]
    total = round(total, 4)
    shown_name = name
    if not unit and total > 1:  # "3 eggs", not "3 egg"
        head, _, last = name.rpartition(" ")
        shown_name = f"{head} {pluralize(last)}".strip()
    display = f"{format_quantity(total, unit, first['measurable'])} {shown_name}"
    return {"name": name, "quantity": total,
            "unit": UNIT_DISPLAY.get(unit, unit) if first["measurable"] else unit,
            "display": display, "recipes": recipes, "sources": sources,
            "amount_known": True, "unknown_amount_recipes": [], "hints": []}


def _unknown_amount_note(recipes):
    return "amount not given" if len(recipes) == 1 else "amount varies by recipe"


def _hints(entries):
    """Raw Food.com amounts (no units in the data), only where the recipe's lists line up."""
    return [{"recipe": e["recipe"], "amount": e["hint"]} for e in entries if e.get("hint")]


def build_grocery_list(recipes, parse_fn):
    """Merge ingredients across recipes.

    Each recipe is {"name", "lines": [str] | None, "ingredient_names": [str],
    "amount_hints": [str | None] | None}. Pasted recipes have `lines`, which are parsed and
    merged with real quantities. `lines=None` (Food.com) merges by name only; `amount_hints`
    are its raw unitless amounts, aligned with `ingredient_names`, shown as hints, never summed.

    Returns {"items": [...], "unmerged": [...]}. Same-name entries with compatible units are
    summed into one item. A quantity-less line ("salt to taste") folds into the measured item
    when one exists. Name-only entries join the item for that name, which then has
    amount_known=False and lists them in unknown_amount_recipes. Same-name entries that can't
    be combined (volume vs. weight) go to "unmerged" with a reason.
    """
    by_name, unmerged = {}, []
    for recipe in recipes:
        entries, unparsed = _parse_lines(recipe, parse_fn)
        unmerged.extend(unparsed)
        for e in entries:
            by_name.setdefault(e["name"], []).append(e)

    items = []
    for name, entries in by_name.items():
        quantified = [e for e in entries if e["quantity"] is not None]
        to_taste = [e for e in entries if e["quantity"] is None and not e["name_only"]]
        name_only = [e for e in entries if e["name_only"]]

        name_only_recipes = sorted({e["recipe"] for e in name_only})
        if not quantified:
            display = f"{name} ({_unknown_amount_note(name_only_recipes)})" if name_only else name
            items.append({"name": name, "quantity": None, "unit": "", "display": display,
                          "recipes": sorted({e["recipe"] for e in entries}),
                          "sources": [e["source"] for e in entries],
                          "amount_known": False, "unknown_amount_recipes": name_only_recipes,
                          "hints": _hints(name_only)})
            continue

        groups = {}
        for e in quantified:
            groups.setdefault(_group_key(e), []).append(e)
        merged = [_merge_group(name, g) for g in groups.values()]
        if to_taste:  # the measured amount covers "to taste"; keep its recipes and sources
            merged[0]["recipes"] = sorted(set(merged[0]["recipes"]) | {e["recipe"] for e in to_taste})
            merged[0]["sources"] += [e["source"] for e in to_taste]
        if len(merged) == 1:
            item = merged[0]
            if name_only:  # measured amount from pasted recipes, plus Food.com recipes without one
                others = len(name_only_recipes)
                item.update(
                    display=f"{item['display']} + more for {others} other recipe{'s' if others > 1 else ''} "
                            f"(amount not given)",
                    recipes=sorted(set(item["recipes"]) | set(name_only_recipes)),
                    sources=item["sources"] + [e["source"] for e in name_only],
                    amount_known=False, unknown_amount_recipes=name_only_recipes, hints=_hints(name_only),
                )
            items.append(item)
        else:
            for m in merged:
                unmerged.append({**m, "reason": "units can't be combined (e.g. volume vs. weight)"})
            if name_only:
                unmerged.append({"name": name, "display": f"{name} ({_unknown_amount_note(name_only_recipes)})",
                                 "recipes": name_only_recipes, "sources": [e["source"] for e in name_only],
                                 "amount_known": False, "unknown_amount_recipes": name_only_recipes,
                                 "hints": _hints(name_only), "reason": "amount not given in recipe"})

    items.sort(key=lambda i: i["name"])
    unmerged.sort(key=lambda i: i["name"])
    return {"items": items, "unmerged": unmerged}


# --- Diet / exclusion checks (keyword heuristic) ---

MEAT_FISH = [
    "chicken", "beef", "pork", "bacon", "ham", "sausage", "turkey", "lamb", "veal", "duck",
    "prosciutto", "pancetta", "salami", "pepperoni", "chorizo", "venison", "lard", "gelatin",
    "steak", "meat", "meatball", "hot dog", "fish", "salmon", "tuna", "cod", "tilapia", "halibut",
    "anchovy", "sardine", "shrimp", "prawn", "crab", "lobster", "clam", "mussel", "oyster",
    "scallop", "fish sauce", "worcestershire sauce",
]
DAIRY = [
    "milk", "butter", "cheese", "cream", "yogurt", "ghee", "buttermilk", "parmesan",
    "mozzarella", "cheddar", "ricotta", "feta", "sour cream", "whey", "half-and-half",
]
EGG_HONEY = ["egg", "egg yolk", "egg white", "honey", "mayonnaise"]
GLUTEN = [
    "flour", "wheat", "bread", "breadcrumb", "bread crumb", "pasta", "spaghetti", "noodle",
    "macaroni", "penne", "lasagna", "fettuccine", "linguine", "orzo", "couscous", "barley",
    "rye", "semolina", "bulgur", "farro", "seitan", "cracker", "tortilla", "soy sauce", "beer",
    "panko", "pita", "bun", "roll", "crouton",
]
NUTS = [
    "nut", "almond", "walnut", "pecan", "cashew", "pistachio", "hazelnut", "filbert",
    "macadamia", "pine nut", "brazil nut", "chestnut", "peanut", "praline", "marzipan",
    "nutella", "gianduja", "nougat", "pesto",
]
HIGH_CARB = [
    "sugar", "brown sugar", "honey", "maple syrup", "flour", "rice", "pasta", "spaghetti",
    "noodle", "bread", "potato", "corn", "oat", "tortilla", "cracker",
]
# Phrases that contain a keyword but are fine for that diet ("peanut butter" isn't dairy).
DAIRY_SAFE = [
    "peanut butter", "almond butter", "cashew butter", "nut butter", "apple butter",
    "coconut milk", "almond milk", "soy milk", "oat milk", "rice milk", "coconut cream",
    "cream of tartar", "vegan", "dairy-free", "dairy free",
]
MEAT_SAFE = ["vegetarian", "vegan", "meatless", "plant-based"]
GLUTEN_SAFE = [
    "almond flour", "rice flour", "coconut flour", "chickpea flour", "corn tortilla",
    "rice noodle", "oat flour", "gluten-free", "gluten free",  # oat flour: allowed with a DIET_NOTES warning
]
NUT_SAFE = ["water chestnut", "nut-free", "nut free"]  # nutmeg/coconut/butternut don't match \bnut\b
CARB_SAFE = ["cauliflower rice", "almond flour", "coconut flour", "sugar-free", "sugar free"]

DIET_KEYWORDS = {  # diet -> (off-limits keywords, safe phrases)
    "vegetarian": (MEAT_FISH, MEAT_SAFE),
    "vegan": (MEAT_FISH + DAIRY + EGG_HONEY, MEAT_SAFE + DAIRY_SAFE),
    "dairy-free": (DAIRY, DAIRY_SAFE),
    "gluten-free": (GLUTEN, GLUTEN_SAFE),
    "nut-free": (NUTS, NUT_SAFE),
    "keto": (HIGH_CARB, CARB_SAFE),
    "low-carb": (HIGH_CARB, CARB_SAFE),
}
DIET_ALIASES = {
    "veg": "vegetarian", "veggie": "vegetarian", "plant-based": "vegan", "plant based": "vegan",
    "gluten free": "gluten-free", "gf": "gluten-free", "dairy free": "dairy-free",
    "no dairy": "dairy-free", "lactose-free": "dairy-free", "low carb": "low-carb",
    "nut free": "nut-free", "no nuts": "nut-free", "no nut": "nut-free", "nut allergy": "nut-free",
    "tree nut free": "nut-free", "tree-nut-free": "nut-free",
}


def canonical_diet(diet):
    """Map free-form diet text to a key of DIET_KEYWORDS, or None if we don't know it."""
    if not diet:
        return None
    d = diet.strip().lower()
    d = DIET_ALIASES.get(d, d)
    return d if d in DIET_KEYWORDS else None


def _contains_phrase(name, phrase):
    return re.search(rf"\b{re.escape(phrase)}\b", name) is not None


def conflicts_with_diet(name, diet):
    """True if a (normalized) ingredient name looks off-limits for the diet."""
    diet = canonical_diet(diet)
    if diet is None:
        return False
    keywords, safe = DIET_KEYWORDS[diet]
    name = normalize_name(name)
    if any(_contains_phrase(name, phrase) for phrase in safe):
        return False
    return any(_contains_phrase(name, kw) for kw in keywords)


# Allowed, but worth a warning: oats are gluten-free yet usually processed alongside wheat.
DIET_NOTES = {  # diet -> [(keywords, note)]
    "gluten-free": [(["oat", "oatmeal"], "use oats labeled gluten-free")],
}


def diet_note(name, diet):
    """A caution for an ingredient the diet allows with care (oats when gluten-free), else None."""
    diet = canonical_diet(diet)
    name = normalize_name(name)
    for keywords, note in DIET_NOTES.get(diet, []):
        if any(_contains_phrase(name, kw) for kw in keywords):
            return note
    return None


def add_diet_notes(items, diet):
    """Set item["diet_note"] on items that need one (e.g. oats on a gluten-free list)."""
    for item in items:
        note = diet_note(item["name"], diet)
        if note:
            item["diet_note"] = note
    return items


def ingredient_matches(name, term):
    """'cremini mushrooms' matches 'mushroom' (whole words, singular/plural-insensitive, synonyms)."""
    term = normalize_name(term)
    return bool(term) and _contains_phrase(normalize_name(name), term)


def flag_for_substitution(items, diet=None, exclude=()):
    """Return [(item, constraint)] for items that break the diet or an exclusion."""
    flagged = []
    for item in items:
        hit = next((ex for ex in exclude if ingredient_matches(item["name"], ex)), None)
        if hit:
            flagged.append((item, f"no {normalize_name(hit)}"))
        elif conflicts_with_diet(item["name"], diet):
            flagged.append((item, canonical_diet(diet)))
    return flagged


def apply_substitutions(flagged, substitute_fn, max_calls=10, max_workers=5):
    """Run substitute_fn (signature of api.get_substitute) for flagged items, concurrently.

    Attaches the result to each item as item["substitute"] and returns a summary list.
    Items past max_calls are reported as skipped instead of silently dropped.
    """
    to_run, skipped = flagged[:max_calls], flagged[max_calls:]

    def run(pair):
        item, constraint = pair
        amount = format_amount(item["quantity"]) if item.get("quantity") is not None else ""
        try:
            result = substitute_fn(item["name"], amount, item.get("unit", ""), constraint)
        except Exception as e:  # one failed call shouldn't sink the whole list
            result = {"found": False, "reason": f"Substitution error: {e}"}
        item["substitute"] = result
        return {"original": item["name"], "constraint": constraint, "result": result}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        summary = list(pool.map(run, to_run))
    for item, constraint in skipped:
        summary.append({"original": item["name"], "constraint": constraint,
                        "result": {"found": False, "reason": "Skipped: too many items to substitute at once"}})
    return summary
