import pytest

from train_bert_colab import apply_label_mode, dedupe, entity_report, fix_labels, get_spans, split, token_report


def sent(text, tags):
    return (text.split(), tags.split())


def test_dedupe_is_case_insensitive_and_keeps_first():
    a = sent("1 cup Salt", "B-AMT B-UNIT B-NAME")
    b = sent("1 cup salt", "B-AMT B-UNIT O")
    c = sent("2 eggs", "B-AMT B-NAME")
    assert dedupe([a, b, c]) == [a, c]


def test_split_is_deterministic_disjoint_and_complete():
    sentences = [sent(f"{i} cup flour", "B-AMT B-UNIT B-NAME") for i in range(100)]
    s1, s2 = split(sentences, seed=42), split(sentences, seed=42)
    assert s1 == s2
    assert [len(s1[k]) for k in ("train", "val", "test")] == [80, 10, 10]
    keys = {k: {tuple(t) for t, _ in v} for k, v in s1.items()}
    assert not (keys["train"] & keys["test"]) and not (keys["val"] & keys["test"])
    assert len(keys["train"] | keys["val"] | keys["test"]) == 100


def test_label_modes_share_the_same_split():
    sentences = [sent(f"{i} cup a flour", "B-AMT B-UNIT B-NAME I-NAME") for i in range(50)]
    splits = split(sentences)
    fixed = apply_label_mode(splits["test"], True)
    assert [t for t, _ in fixed] == [t for t, _ in splits["test"]]


@pytest.mark.parametrize("text, tags, expected", [
    # substring-matching noise: function words / punctuation labeled NAME
    ("1 cup a flour", "B-AMT B-UNIT B-NAME I-NAME", "B-AMT B-UNIT O B-NAME"),
    ("1 ( 15 ounce ) can beans", "B-AMT B-NAME B-AMT B-UNIT B-NAME B-UNIT B-NAME",
     "B-AMT O B-AMT B-UNIT O B-UNIT B-NAME"),
    # function words inside a name stay
    ("1 tsp cream of tartar", "B-AMT B-UNIT B-NAME I-NAME I-NAME", "B-AMT B-UNIT B-NAME I-NAME I-NAME"),
    ("salt , to taste", "B-NAME O O B-DESC", "B-NAME O O B-DESC"),
    # missing I- tags
    ("1 1/2 cups flour", "B-AMT B-AMT B-UNIT B-NAME", "B-AMT I-AMT B-UNIT B-NAME"),
    ("2 - 3 tomatoes", "B-AMT O B-AMT B-NAME", "B-AMT I-AMT I-AMT B-NAME"),
    ("2 to 3 tomatoes", "B-AMT O B-AMT B-NAME", "B-AMT I-AMT I-AMT B-NAME"),
    ("2 fluid ounces milk", "B-AMT B-DESC B-UNIT B-NAME", "B-AMT B-UNIT I-UNIT B-NAME"),
    ("1 cup finely chopped onion", "B-AMT B-UNIT B-DESC B-DESC B-NAME", "B-AMT B-UNIT B-DESC I-DESC B-NAME"),
    # two separate numbers stay separate entities
    ("1 10-ounce package", "B-AMT B-DESC B-UNIT", "B-AMT B-DESC B-UNIT"),
])
def test_fix_labels(text, tags, expected):
    tokens, tag_list = sent(text, tags)
    assert fix_labels(tokens, tag_list) == expected.split()


def test_fix_labels_output_is_valid_bio():
    tokens, tags = sent("a banana , diced", "B-NAME I-NAME O B-DESC")
    fixed = fix_labels(tokens, tags)
    assert fixed == ["O", "B-NAME", "O", "B-DESC"]  # I-NAME after O was promoted to B-NAME


def test_entity_report_is_span_level():
    y_true = [["B-AMT", "I-AMT", "B-UNIT", "B-NAME"]]
    y_pred = [["B-AMT", "B-AMT", "B-UNIT", "B-NAME"]]  # amount split into two spans
    r = entity_report(y_true, y_pred)
    assert r["AMT"]["recall"] == 0.0 and r["AMT"]["precision"] == 0.0
    assert r["UNIT"]["f1-score"] == 1.0 and r["NAME"]["f1-score"] == 1.0


def test_token_report_ignores_span_convention():
    y_true = [["B-AMT", "I-AMT", "B-UNIT", "O"]]
    y_pred = [["B-AMT", "B-AMT", "B-UNIT", "B-NAME"]]
    r = token_report(y_true, y_pred)
    assert r["AMT"]["f1-score"] == 1.0
    assert r["NAME"]["precision"] == 0.0 and r["NAME"]["support"] == 0


@pytest.mark.parametrize("tags, spans", [
    ("B-AMT I-AMT B-UNIT", [("AMT", 0, 1), ("UNIT", 2, 2)]),
    ("B-DESC B-DESC", [("DESC", 0, 0), ("DESC", 1, 1)]),  # B- always starts a new span
    ("O I-NAME I-NAME", [("NAME", 1, 2)]),  # stray I- starts a span (conlleval rule)
    ("B-NAME I-DESC", [("NAME", 0, 0), ("DESC", 1, 1)]),  # type change ends the span
    ("O O", []),
])
def test_get_spans_follows_conlleval(tags, spans):
    assert get_spans(tags.split()) == spans


def test_entity_report_averages():
    y_true = [["B-AMT", "B-UNIT", "B-NAME", "I-NAME"]]
    y_pred = [["B-AMT", "B-UNIT", "B-NAME", "O"]]  # NAME span too short
    r = entity_report(y_true, y_pred)
    assert r["micro avg"]["f1-score"] == pytest.approx(2 / 3)
    assert r["macro avg"]["f1-score"] == pytest.approx(2 / 3)
    assert r["weighted avg"]["support"] == 3
