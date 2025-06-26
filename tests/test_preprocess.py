"""Unit tests for dataset splitting and tokenisation helpers."""
from __future__ import annotations

import pytest
from datasets import Dataset

from llm.datasets.preprocess import split_by_column


@pytest.fixture
def dummy_dataset() -> Dataset:
    """Return a small synthetic Dataset with a *split* column."""
    n = 30
    data = {
        "smiles": ["C"] * n,
        "question_id": ["q"] * n,
        "question_template": ["tmpl"] * n,
        "answer": [1] * n,
        "assistant_template": ["ans {answer}"] * n,
        "system_prompt": ["sys"] * n,
        "metadata": [{}] * n,
        "split": ["train"] * 20 + ["valid"] * 5 + ["test"] * 5,
    }
    return Dataset.from_dict(data)


def test_split_by_column_removes_split(dummy_dataset):
    dsdict = split_by_column(dummy_dataset)
    assert set(dsdict.keys()) == {"train", "valid", "test"}
    # verify column removed
    for ds in dsdict.values():
        assert "split" not in ds.column_names
        # counts preserved
    assert len(dsdict["train"]) == 20
    assert len(dsdict["valid"]) == 5
    assert len(dsdict["test"]) == 5

