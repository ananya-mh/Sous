"""How the BERT pipeline's subword predictions become parse_recipe_bert fields."""
import ast
from pathlib import Path

import pytest
import torch
from transformers import BertConfig, BertForTokenClassification, BertTokenizerFast, TokenClassificationPipeline
from transformers.modeling_outputs import TokenClassifierOutput

from grocery_list import parse_quantity
from train_bert_colab import LABEL2ID, LABEL_LIST, parse_like_api

REPO = Path(__file__).resolve().parent.parent

# "minced" is two subwords: "min" + "##ced". Training labels only the first subword, so the
# prediction for "##ced" is untrained; here it's deliberately wrong (B-NAME).
SUBWORD_PREDICTIONS = {
    "3": "B-AMT", "cloves": "B-UNIT", "garlic": "B-NAME", ",": "O", "min": "B-DESC", "##ced": "B-NAME",
}


class FakeTokenModel(BertForTokenClassification):
    """Returns fixed logits per vocabulary token instead of running BERT."""

    def __init__(self, config, id_to_label):
        super().__init__(config)
        self.id_to_label = id_to_label

    def forward(self, input_ids=None, **kwargs):
        logits = torch.full((*input_ids.shape, len(LABEL_LIST)), -10.0)
        for b, row in enumerate(input_ids.tolist()):
            for i, token_id in enumerate(row):
                logits[b, i, LABEL2ID[self.id_to_label.get(token_id, "O")]] = 10.0
        return TokenClassifierOutput(logits=logits)


@pytest.fixture
def fake_pipeline(tmp_path):
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *SUBWORD_PREDICTIONS]
    (tmp_path / "vocab.txt").write_text("\n".join(vocab) + "\n")
    tokenizer = BertTokenizerFast(vocab_file=str(tmp_path / "vocab.txt"))
    config = BertConfig(vocab_size=len(vocab), hidden_size=8, num_hidden_layers=1, num_attention_heads=1,
                        intermediate_size=8, num_labels=len(LABEL_LIST),
                        id2label=dict(enumerate(LABEL_LIST)), label2id=LABEL2ID)
    model = FakeTokenModel(config, {vocab.index(t): label for t, label in SUBWORD_PREDICTIONS.items()})

    def make(strategy):
        return TokenClassificationPipeline(model=model, tokenizer=tokenizer, aggregation_strategy=strategy)
    return make


def test_first_keeps_a_multi_subword_word_in_one_field(fake_pipeline):
    parsed = parse_like_api(fake_pipeline("first"), "3 cloves garlic, minced")
    assert parsed == {"amount": "3", "unit": "cloves", "item": "garlic", "descriptor": "minced"}


def test_simple_splits_the_word_on_the_untrained_subword(fake_pipeline):
    # Why api.py doesn't use "simple": each subword keeps its own prediction.
    parsed = parse_like_api(fake_pipeline("simple"), "3 cloves garlic, minced")
    assert parsed["descriptor"] == "min"
    assert parsed["item"] == "garlic ced"


@pytest.mark.parametrize("path", ["scripts/api.py", "scripts/inference.py", "scripts/train_bert_colab.py"])
def test_pipelines_use_first_aggregation(path):
    tree = ast.parse((REPO / path).read_text())
    strategies = [kw.value.value for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and getattr(node.func, "id", None) == "pipeline"
                  for kw in node.keywords if kw.arg == "aggregation_strategy"]
    assert strategies == ["first"]


FIXED_MODEL = REPO / "fixed/model"


@pytest.mark.skipif(not (FIXED_MODEL / "model.safetensors").exists(), reason="fixed model not present")
def test_fixed_model_amount_for_fraction_needs_no_fallback():
    from transformers import pipeline
    nlp = pipeline("token-classification", model=str(FIXED_MODEL), aggregation_strategy="first")
    entities = nlp("1/2 teaspoon salt")
    # Every piece of "1/2" is AMT and nothing else is. The pipeline splits "1/2" into the words
    # "1", "/", "2", and the model tags each B-AMT (three entities), but together they read as 1/2.
    assert [(e["entity_group"], e["word"]) for e in entities] == [
        ("AMT", "1"), ("AMT", "/"), ("AMT", "2"), ("UNIT", "teaspoon"), ("NAME", "salt"),
    ]
    assert parse_quantity(parse_like_api(nlp, "1/2 teaspoon salt")["amount"]) == 0.5
