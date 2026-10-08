import pytest

from evaluate_gold import score
from gold_set import read_gold, write_gold

SENTENCES = [
    {"id": "nyt-0001", "tokens": ["1", "1/2", "cups", "flour"], "tags": ["B-AMT", "I-AMT", "B-UNIT", "B-NAME"]},
    {"id": "nyt-0002", "tokens": ["salt", ",", "to", "taste"], "tags": ["B-NAME", "O", "O", "B-DESC"]},
]


def test_round_trip(tmp_path):
    path = tmp_path / "nyt.tsv"
    write_gold(path, SENTENCES, "nyt")
    gold = read_gold(path)
    assert [(s["id"], s["tokens"], s["tags"], s["reviewed"]) for s in gold] == [
        (s["id"], s["tokens"], s["tags"], False) for s in SENTENCES
    ]


def test_refuses_to_overwrite(tmp_path):
    path = tmp_path / "nyt.tsv"
    write_gold(path, SENTENCES, "nyt")
    with pytest.raises(FileExistsError):
        write_gold(path, SENTENCES, "nyt")


def test_reads_hand_edits(tmp_path):
    path = tmp_path / "nyt.tsv"
    write_gold(path, SENTENCES, "nyt")
    first = "# text: 1 1/2 cups flour\n# reviewed: no"
    text = path.read_text().replace(first, first.replace("no", "yes"))
    text = text.replace("taste", "taste ", 1)  # trailing whitespace from an editor is fine
    path.write_text(text + "\n\n")
    gold = read_gold(path)
    assert [s["reviewed"] for s in gold] == [True, False]


@pytest.mark.parametrize("bad_line, message", [
    ("flour\tB-NAMEE", "unknown label 'B-NAMEE'"),
    ("flour", "expected 'token<TAB>label'"),
])
def test_validation_errors_point_at_the_line(tmp_path, bad_line, message):
    path = tmp_path / "bad.tsv"
    path.write_text(f"# id: x\n# reviewed: yes\n1\tB-AMT\n{bad_line}\n")
    with pytest.raises(ValueError, match=rf"bad.tsv:4: {message}"):
        read_gold(path)


def test_score_counts_reviewed_and_reports_both_levels():
    gold = [dict(s, reviewed=i == 0) for i, s in enumerate(SENTENCES)]
    predictions = [["B-AMT", "B-AMT", "B-UNIT", "B-NAME"], ["B-NAME", "O", "O", "B-DESC"]]
    result = score(gold, predictions)
    assert (result["sentences"], result["reviewed"]) == (2, 1)
    assert result["entity_level"]["AMT"]["f1-score"] == 0.0  # split mixed number is a wrong span
    assert result["token_level"]["AMT"]["f1-score"] == 1.0
    assert result["entity_level"]["NAME"]["f1-score"] == 1.0
