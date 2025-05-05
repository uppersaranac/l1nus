import pytest
import numpy as np
import torch
from unittest.mock import MagicMock
from transformers import AutoTokenizer
from llm_apis import build_train_batch, build_eval_batch, _norm, compute_metrics_closure, do_generation

@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token
    return tok

@pytest.fixture
def toy_data():
    return ["CCO", "CC(=O)O"], ["ethanol", "acetic acid"]

def test_build_train_batch(tokenizer, toy_data):
    smiles, iupac = toy_data
    max_len = 32
    batch = build_train_batch(tokenizer, smiles, iupac, max_len=max_len)

    input_ids = batch["input_ids"]
    labels = batch["labels"]

    assert len(input_ids) == len(smiles)
    assert all(len(seq) == max_len for seq in input_ids)
    assert all(len(seq) == max_len for seq in labels)

    for inp, lab in zip(input_ids, labels):
        padding_len = sum(1 for tok in inp if tok == tokenizer.pad_token_id)
        assert all(tok == -100 for tok in lab[:padding_len])
        for i in range(max_len):
            if lab[i] != -100:
                assert lab[i] == inp[i]

def test_build_eval_batch(tokenizer, toy_data):
    smiles, iupac = toy_data
    max_prompt_len = 32
    max_label_len = 16
    batch = build_eval_batch(tokenizer, smiles, iupac, max_prompt_len, max_label_len)

    input_ids = batch["input_ids"]
    labels = batch["labels"]

    assert len(input_ids) == len(smiles)
    assert all(len(seq) == max_prompt_len for seq in input_ids)
    assert all(len(seq) == max_prompt_len for seq in labels)

    for label in labels:
        assert all(tok == -100 for tok in label[:max_prompt_len - max_label_len])
        assert any(tok != -100 for tok in label[max_prompt_len - max_label_len:])

def test_norm():
    assert _norm("It is ethanol.") == "ethanol"
    assert _norm(" ethanol ") == "ethanol"
    assert _norm("It is acetic acid") == "acetic acid"

def test_compute_metrics_closure_exact_match(tokenizer):
    compute_metrics = compute_metrics_closure(tokenizer)
    ids = tokenizer("ethanol", add_special_tokens=False)["input_ids"]
    pred = np.array([ids])
    label = np.array([[i if i != tokenizer.pad_token_id else -100 for i in ids]])
    result = compute_metrics((pred, label))
    assert "exact_match" in result
    assert result["exact_match"] == 1.0

def test_do_generation_mock(tokenizer):
    dummy_input_ids = torch.full((2, 10), tokenizer.pad_token_id)
    dummy_attention_mask = torch.ones_like(dummy_input_ids)

    class DummyDataset:
        def set_format(self, type=None, columns=None):
            pass
        def __getitem__(self, key):
            if key == "input_ids":
                return dummy_input_ids
            elif key == "attention_mask":
                return dummy_attention_mask

    model = MagicMock()
    model.device = torch.device("cpu")
    model.generate = MagicMock(return_value=torch.tensor([[1, 2, 3], [4, 5, 6]]))

    preds = do_generation(num_beams=1, max_new_tokens=5,
                          tokenizer=tokenizer, model=model,
                          data=DummyDataset())
    assert isinstance(preds, torch.Tensor)
    assert preds.shape == (2, 3)
    model.generate.assert_called_once()