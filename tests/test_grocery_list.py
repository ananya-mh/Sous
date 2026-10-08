import pytest

from grocery_list import (
    apply_substitutions, build_grocery_list, conflicts_with_diet, flag_for_substitution,
    format_amount, ingredient_matches, normalize_name, parse_quantity, simple_parse,
)


def fake_parser(table):
    """parse_fn that returns canned parses, so merge tests don't depend on any parser."""
    def parse(text):
        amount, unit, item = table[text]
        return {"amount": amount, "unit": unit, "item": item, "descriptor": ""}
    return parse


def recipe(name, lines=None, names=()):
    return {"name": name, "lines": lines, "ingredient_names": list(names)}


# --- normalize_name ---

@pytest.mark.parametrize("raw, expected", [
    ("Tomatoes", "tomato"),
    ("diced tomatoes", "tomato"),
    ("2 large eggs", "egg"),
    ("onion, finely chopped", "onion"),
    ("fresh basil leaves", "basil leaf"),
    ("cherry tomatoes (halved)", "cherry tomato"),
    ("berries", "berry"),
    ("peaches", "peach"),
    ("molasses", "molasses"),
    ("asparagus", "asparagus"),
    ("garlic cloves", "garlic clove"),
    ("cream of tartar", "cream of tartar"),
    ("scallions", "green onion"),
    ("Confectioners' sugar", "powdered sugar"),
    ("extra virgin olive oil", "olive oil"),
    ("salt to taste", "salt"),
    ("parsley for garnish", "parsley"),
    ("vegetable oil, for frying", "vegetable oil"),
    ("packed brown sugar", "brown sugar"),
])
def test_normalize_name(raw, expected):
    assert normalize_name(raw) == expected


# --- parse_quantity ---

@pytest.mark.parametrize("text, expected", [
    ("2", 2.0),
    ("1/2", 0.5),
    ("1 1/2", 1.5),
    ("1 / 4", 0.25),  # BERT detokenization
    ("½", 0.5),
    ("1½", 1.5),
    ("1.5", 1.5),
    ("2-3", 3.0),
    ("2 to 3", 3.0),
    ("1-1/2", 1.5),
    ("a", 1.0),
    ("", None),
    ("1 14", None),  # "1 (14 oz) can" -> ambiguous, don't guess
    ("some", None),
])
def test_parse_quantity(text, expected):
    assert parse_quantity(text) == expected


def test_format_amount():
    assert format_amount(1.5) == "1 1/2"
    assert format_amount(0.3333) == "1/3"
    assert format_amount(2.0) == "2"
    assert format_amount(0.01) == "0.01"


# --- merging ---

def test_sums_same_unit_across_recipes():
    parse = fake_parser({"1 cup rice": ("1", "cup", "rice"), "2 cups rice": ("2", "cups", "rice")})
    out = build_grocery_list([recipe("A", ["1 cup rice"]), recipe("B", ["2 cups rice"])], parse)
    assert out["unmerged"] == []
    [item] = out["items"]
    assert item["name"] == "rice"
    assert item["quantity"] == 3
    assert item["display"] == "3 cups rice"
    assert item["recipes"] == ["A", "B"]


def test_converts_compatible_units():
    parse = fake_parser({"1 cup milk": ("1", "cup", "milk"), "4 tbsp milk": ("4", "tbsp", "milk")})
    out = build_grocery_list([recipe("A", ["1 cup milk", "4 tbsp milk"])], parse)
    [item] = out["items"]
    assert item["unit"] == "cup"  # reported in the larger unit
    assert item["quantity"] == pytest.approx(1.25)


@pytest.mark.parametrize("amounts, expected", [
    # your cases
    ([("2", "tbsp"), ("1/4", "cup")], "6 tbsp"),  # 3/8 cup isn't a measuring cup
    ([("1/2", "cup"), ("1/4", "cup")], "3/4 cup"),
    ([("1", "cup"), ("4", "tbsp")], "1 1/4 cups"),
    ([("1", "tsp"), ("1", "tsp")], "2 tsp"),
    ([("1", "tsp"), ("2", "tsp")], "1 tbsp"),
    ([("8", "oz"), ("8", "oz")], "1 lb"),
    ([("200", "g"), ("300", "g")], "500 g"),
    # edges
    ([("5", "tbsp"), ("1", "tsp")], "1/3 cup"),  # exactly 1/3 cup
    ([("2", "tbsp"), ("2", "tbsp")], "1/4 cup"),
    ([("2", "tbsp"), ("1", "tbsp")], "3 tbsp"),
    ([("1", "tsp")], "1 tsp"),
    ([("1", "cup")], "1 cup"),
    ([("2", "cups"), ("1", "pint")], "4 cups"),
    ([("4", "oz"), ("4", "oz")], "8 oz"),
    ([("1", "lb"), ("4", "oz")], "1 1/4 lb"),
    ([("600", "g"), ("500", "g")], "1.1 kg"),
    ([("1", "kg"), ("500", "g")], "1 1/2 kg"),
    ([("600", "ml"), ("600", "ml")], "1.2 l"),
    ([("250", "ml"), ("250", "ml")], "500 ml"),
    ([("100", "g"), ("4", "oz")], "7.53 oz"),  # mixed metric + US -> US units
])
def test_display_units(amounts, expected):
    lines = [f"{a} {u} sugar" for a, u in amounts]
    parse = fake_parser({line: (a, u, "sugar") for line, (a, u) in zip(lines, amounts)})
    out = build_grocery_list([recipe(f"R{i}", [line]) for i, line in enumerate(lines)], parse)
    assert out["unmerged"] == []
    assert [i["display"] for i in out["items"]] == [f"{expected} sugar"]


def test_merges_names_after_normalization():
    parse = fake_parser({
        "2 diced tomatoes": ("2", "", "diced tomatoes"),
        "1 tomato, chopped": ("1", "", "tomato, chopped"),
    })
    out = build_grocery_list([recipe("A", ["2 diced tomatoes"]), recipe("B", ["1 tomato, chopped"])], parse)
    [item] = out["items"]
    assert item["display"] == "3 tomatoes"


def test_volume_and_weight_are_listed_separately():
    parse = fake_parser({"2 cups flour": ("2", "cups", "flour"), "200 g flour": ("200", "g", "flour")})
    out = build_grocery_list([recipe("A", ["2 cups flour"]), recipe("B", ["200 g flour"])], parse)
    assert out["items"] == []
    assert sorted(u["display"] for u in out["unmerged"]) == ["2 cups flour", "200 g flour"]
    assert all("can't be combined" in u["reason"] for u in out["unmerged"])


def test_count_units_merge_only_with_same_unit():
    parse = fake_parser({
        "2 cloves garlic": ("2", "cloves", "garlic"),
        "1 clove garlic": ("1", "clove", "garlic"),
        "1 head garlic": ("1", "head", "garlic"),
    })
    out = build_grocery_list([recipe("A", ["2 cloves garlic", "1 clove garlic"])], parse)
    assert out["items"][0]["display"] == "3 cloves garlic"

    out = build_grocery_list([recipe("A", ["2 cloves garlic"]), recipe("B", ["1 head garlic"])], parse)
    assert out["items"] == []
    assert len(out["unmerged"]) == 2


def test_to_taste_folds_into_measured_item():
    parse = fake_parser({"1 tsp salt": ("1", "tsp", "salt"), "salt to taste": ("", "", "salt")})
    out = build_grocery_list([recipe("A", ["1 tsp salt"]), recipe("B", ["salt to taste"])], parse)
    assert out["unmerged"] == []
    [item] = out["items"]
    assert item["display"] == "1 tsp salt"
    assert item["recipes"] == ["A", "B"]
    assert item["sources"] == ["1 tsp salt", "salt to taste"]


def test_to_taste_alone_is_an_item_without_quantity():
    parse = fake_parser({"pepper to taste": ("", "", "pepper")})
    out = build_grocery_list([recipe("A", ["pepper to taste"])], parse)
    assert [(i["name"], i["quantity"]) for i in out["items"]] == [("pepper", None)]


def test_synonyms_merge():
    parse = fake_parser({
        "2 scallions": ("2", "", "scallions"),
        "3 green onions": ("3", "", "green onions"),
        "1 cup all-purpose flour": ("1", "cup", "all-purpose flour"),
        "1 cup flour": ("1", "cup", "flour"),
    })
    out = build_grocery_list([recipe("A", ["2 scallions", "1 cup all-purpose flour"]),
                              recipe("B", ["3 green onions", "1 cup flour"])], parse)
    assert [i["display"] for i in out["items"]] == ["2 cups flour", "5 green onions"]


def test_count_unit_inside_name_merges_with_unit_form():
    parse = fake_parser({"2 garlic cloves": ("2", "", "garlic cloves"), "3 cloves garlic": ("3", "cloves", "garlic")})
    out = build_grocery_list([recipe("A", ["2 garlic cloves"]), recipe("B", ["3 cloves garlic"])], parse)
    assert [i["display"] for i in out["items"]] == ["5 cloves garlic"]


def test_unreadable_quantity_or_name_goes_to_unmerged():
    parse = fake_parser({"one or two (14 oz) cans beans": ("1 14", "oz can", "beans"), "???": ("", "", "")})
    out = build_grocery_list([recipe("A", ["one or two (14 oz) cans beans", "???"])], parse)
    assert out["items"] == []
    assert sorted(u["reason"] for u in out["unmerged"]) == [
        "couldn't identify the ingredient", "couldn't read the quantity",
    ]


@pytest.mark.parametrize("line, bert_amount, expected", [
    ("1/2 cup sugar", "1 2", "1/2 cup sugar"),  # what the real model returns for fractions
    ("2 1/2 cups sugar", "2 1 2", "2 1/2 cups sugar"),
    ("2 to 3 cups sugar", "2 3", "3 cups sugar"),
    ("1 (14 ounce) cup sugar", "1 14", "1 cup sugar"),
])
def test_unreadable_bert_amount_falls_back_to_line(line, bert_amount, expected):
    parse = fake_parser({line: (bert_amount, "cup", "sugar")})
    out = build_grocery_list([recipe("A", [line])], parse)
    assert out["unmerged"] == []
    assert [i["display"] for i in out["items"]] == [expected]


def test_name_only_recipe_merges_by_name():
    parse = fake_parser({"2 onions": ("2", "", "onions")})
    out = build_grocery_list([
        recipe("A", lines=None, names=["onion", "Carrots"]),
        recipe("B", lines=None, names=["onions"]),
    ], parse)
    by_name = {i["name"]: i for i in out["items"]}
    assert by_name["onion"]["quantity"] is None
    assert by_name["onion"]["recipes"] == ["A", "B"]
    assert "carrot" in by_name


def test_name_only_joins_the_measured_item():
    parse = fake_parser({"2 onions": ("2", "", "onions")})
    out = build_grocery_list([recipe("A", ["2 onions"]), recipe("B", lines=None, names=["onion"])], parse)
    assert out["unmerged"] == []
    [item] = out["items"]
    assert item["display"] == "2 onions + more for 1 other recipe (amount not given)"
    assert item["quantity"] == 2  # the known part only
    assert (item["amount_known"], item["unknown_amount_recipes"], item["recipes"]) == (False, ["B"], ["A", "B"])


@pytest.mark.parametrize("recipes, expected", [
    ([("A", ["flour"])], "flour (amount not given)"),
    ([("A", ["flour"]), ("B", ["Flour"])], "flour (amount varies by recipe)"),
])
def test_unknown_amount_wording(recipes, expected):
    out = build_grocery_list([recipe(n, lines=None, names=names) for n, names in recipes], fake_parser({}))
    [item] = out["items"]
    assert item["display"] == expected
    assert item["amount_known"] is False and item["quantity"] is None


def test_food_com_garlic_clove_merges_with_pasted_cloves_of_garlic():
    parse = fake_parser({"3 cloves garlic, minced": ("3", "cloves", "garlic")})
    out = build_grocery_list([recipe("Mine", ["3 cloves garlic, minced"]),
                              recipe("Food.com", lines=None, names=["garlic clove"])], parse)
    assert [i["display"] for i in out["items"]] == ["3 cloves garlic + more for 1 other recipe (amount not given)"]


def test_amount_hints_only_where_given():
    out = build_grocery_list([
        {"name": "A", "lines": None, "ingredient_names": ["sugar", "eggs"], "amount_hints": ["1/4", None]},
        {"name": "B", "lines": None, "ingredient_names": ["sugar"], "amount_hints": None},  # lists didn't line up
    ], fake_parser({}))
    by_name = {i["name"]: i for i in out["items"]}
    assert by_name["sugar"]["hints"] == [{"recipe": "A", "amount": "1/4"}]
    assert by_name["egg"]["hints"] == []
    assert by_name["sugar"]["unknown_amount_recipes"] == ["A", "B"]


def test_measured_items_report_amount_known():
    parse = fake_parser({"1 cup rice": ("1", "cup", "rice")})
    [item] = build_grocery_list([recipe("A", ["1 cup rice"])], parse)["items"]
    assert (item["amount_known"], item["unknown_amount_recipes"], item["hints"]) == (True, [], [])


def test_pasted_recipe_with_food_com_recipes():
    out = build_grocery_list([
        recipe("My soup", ["2 cups chopped onions", "1 tablespoon olive oil", "salt to taste"]),
        {"name": "Food.com salad", "lines": None, "ingredient_names": ["onions", "olive oil", "feta cheese"],
         "amount_hints": ["1", "1/4", "4"]},
    ], simple_parse)
    by_name = {i["name"]: i for i in out["items"]}
    assert by_name["onion"]["display"] == "2 cups onion + more for 1 other recipe (amount not given)"
    assert by_name["onion"]["hints"] == [{"recipe": "Food.com salad", "amount": "1"}]
    assert by_name["feta cheese"]["display"] == "feta cheese (amount not given)"
    assert by_name["salt"]["display"] == "salt"  # pasted "to taste": no Food.com wording
    assert out["unmerged"] == []


# --- stand-in parser ---

@pytest.mark.parametrize("line, amount, unit, item", [
    ("1 1/2 cups all-purpose flour", "1 1/2", "cups", "all-purpose flour"),
    ("2 large eggs", "2", "", "eggs"),
    ("3 cloves garlic, minced", "3", "cloves", "garlic"),
    ("1 (14 ounce) can diced tomatoes", "1", "can", "tomatoes"),
    ("2 fluid ounces lime juice", "2", "fluid ounces", "lime juice"),
    ("salt to taste", "", "", "salt to taste"),
])
def test_simple_parse(line, amount, unit, item):
    parsed = simple_parse(line)
    assert set(parsed) == {"amount", "unit", "item", "descriptor"}  # same shape as parse_recipe_bert
    assert (parsed["amount"], parsed["unit"], parsed["item"]) == (amount, unit, item)


def test_simple_parse_end_to_end():
    out = build_grocery_list([
        recipe("A", ["1 cup chopped onion", "2 tablespoons olive oil"]),
        recipe("B", ["1/2 cup diced onions", "1 tbsp olive oil"]),
    ], simple_parse)
    by_name = {i["name"]: i for i in out["items"]}
    assert by_name["onion"]["display"] == "1 1/2 cups onion"
    assert by_name["olive oil"]["display"] == "3 tbsp olive oil"


# --- diet / exclusions / substitutions ---

@pytest.mark.parametrize("name, diet, expected", [
    ("chicken breasts", "vegetarian", True),
    ("vegetable broth", "vegetarian", False),
    ("butter", "vegan", True),
    ("peanut butter", "vegan", False),
    ("butternut squash", "vegan", False),
    ("eggplant", "vegan", False),
    ("large eggs", "vegan", True),
    ("cream of tartar", "dairy-free", False),
    ("all-purpose flour", "gluten-free", True),
    ("almond flour", "gluten-free", False),
    ("corn tortillas", "gluten-free", False),
    ("corn tortillas", "keto", True),  # safe phrases are per-diet
    ("chopped walnuts", "nut-free", True),
    ("pine nuts", "nut-free", True),
    ("peanut butter", "nut-free", True),
    ("basil pesto", "nut-free", True),
    ("nutmeg", "nut-free", False),
    ("butternut squash", "nut-free", False),
    ("water chestnuts", "nut-free", False),
    ("coconut milk", "nut-free", False),  # coconut isn't flagged (documented limitation)
    ("almonds", "no nuts", True),
    ("chicken", "Veg", True),  # aliases
    ("chicken", "pescatarian", False),  # unknown diet: no flags
    ("chicken", None, False),
])
def test_conflicts_with_diet(name, diet, expected):
    assert conflicts_with_diet(name, diet) is expected


def test_ingredient_matches():
    assert ingredient_matches("cremini mushrooms", "mushrooms")
    assert ingredient_matches("mushroom", "Mushrooms")
    assert not ingredient_matches("onion", "mushroom")
    assert not ingredient_matches("walnut", "nut")  # whole words only (documented limitation)


def test_flag_for_substitution_prefers_exclusion():
    items = [{"name": "button mushroom"}, {"name": "bacon"}, {"name": "rice"}]
    flagged = flag_for_substitution(items, diet="vegetarian", exclude=["mushrooms"])
    assert [(i["name"], c) for i, c in flagged] == [("button mushroom", "no mushroom"), ("bacon", "vegetarian")]


def test_apply_substitutions_attaches_results_and_caps_calls():
    calls = []

    def fake_substitute(item, amount, unit, constraint):
        calls.append((item, amount, unit, constraint))
        if item == "boom":
            raise RuntimeError("api down")
        return {"found": True, "substitute_item": f"vegan {item}", "new_amount": amount,
                "new_unit": unit, "reason": "test"}

    items = [{"name": "butter", "quantity": 0.5, "unit": "cup"}, {"name": "boom", "quantity": None},
             {"name": "milk", "quantity": 1, "unit": "cup"}]
    flagged = [(i, "vegan") for i in items]
    summary = apply_substitutions(flagged, fake_substitute, max_calls=2)

    assert ("butter", "1/2", "cup", "vegan") in calls
    assert len(calls) == 2
    assert items[0]["substitute"]["substitute_item"] == "vegan butter"
    assert items[1]["substitute"]["found"] is False  # exception became a not-found result
    assert "Skipped" in summary[2]["result"]["reason"]
    assert "substitute" not in items[2]


@pytest.mark.parametrize("name, diet, note", [
    ("rolled oats", "gluten-free", "use oats labeled gluten-free"),
    ("oatmeal", "gluten free", "use oats labeled gluten-free"),
    ("oat flour", "gluten-free", "use oats labeled gluten-free"),
    ("boat", "gluten-free", None),
    ("rolled oats", "vegan", None),
    ("rolled oats", None, None),
])
def test_diet_note(name, diet, note):
    from grocery_list import diet_note
    assert diet_note(name, diet) == note


def test_oats_are_allowed_but_noted_on_gluten_free_list():
    from grocery_list import add_diet_notes
    items = [{"name": "oat"}, {"name": "oat flour"}, {"name": "flour"}]
    flagged = flag_for_substitution(items, diet="gluten-free")
    assert [item["name"] for item, _ in flagged] == ["flour"]  # oats aren't substituted
    add_diet_notes(items, "gluten-free")
    assert [i.get("diet_note") for i in items] == ["use oats labeled gluten-free"] * 2 + [None]
