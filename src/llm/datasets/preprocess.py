"""Utilities to convert *questions.jsonl* into tokenised HuggingFace Datasets.

This module deliberately depends on the *existing* `llm_apis.process_single_qa`
function so that tokenisation behaviour is identical to the original pipeline
— this is crucial for regression-test parity.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import datasets as hfds
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from llm_apis import process_single_qa  # reuse existing logic

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading & splitting helpers
# ---------------------------------------------------------------------------

def load_questions_jsonl(path: str | Path) -> Dataset:
    """Load the JSONL file produced by *cli_generate.py* into a HF Dataset."""
    ds = hfds.load_dataset("json", data_files=str(path), split="train")
    logger.info("Loaded %d Q-A records from %s", len(ds), path)
    return ds


def split_dataset(ds: Dataset, valid_frac: float = 0.025, test_frac: float = 0.025, seed: int = 42) -> DatasetDict:
    """Split dataset into train/valid/test by fractions (random shuffle)."""
    assert valid_frac + test_frac < 1.0, "Split fractions too large"
    ds = ds.shuffle(seed=seed)
    n = len(ds)
    test_size = int(n * test_frac)
    valid_size = int(n * valid_frac)
    train_size = n - test_size - valid_size
    train_ds, valid_ds, test_ds = hfds.dataset_dict.Dataset.from_dict({}), None, None  # placeholder
    if test_size > 0:
        test_ds = ds.select(range(test_size))
    if valid_size > 0:
        valid_ds = ds.select(range(test_size, test_size + valid_size))
    train_ds = ds.select(range(test_size + valid_size, n))
    splits = {"train": train_ds, "valid": valid_ds, "test": test_ds}
    # Remove None splits
    return DatasetDict({k: v for k, v in splits.items() if v is not None})


# ---------------------------------------------------------------------------
# Split using existing column
# ---------------------------------------------------------------------------

def split_by_column(ds: Dataset, column: str = "split") -> DatasetDict:
    """Return DatasetDict using *column* values ('train' / 'valid' / 'test').

    If a particular split is missing it is omitted from the result.
    The column itself is **removed** from each split to match old behaviour.
    """
    splits = {}
    for name in ["train", "valid", "test"]:
        subset = ds.filter(lambda example: example[column] == name)
        if len(subset):
            splits[name] = subset.remove_columns([column])
    logger.info("Split by column → counts: %s", {k: len(v) for k, v in splits.items()})
    return DatasetDict(splits)


# ---------------------------------------------------------------------------
# Tokenisation helper
# ---------------------------------------------------------------------------

def tokenise_split(
    ds: Dataset,
    tokenizer: AutoTokenizer,
    max_length: int = 4096,
    max_label_len: int | None = 1024,
    is_train: bool = True,
    num_proc: int | None = None,
) -> Dataset:
    """Apply *process_single_qa* to every example in *ds*."""
    logger.info("Tokenising %d examples (is_train=%s)…", len(ds), is_train)
    mapped = ds.map(
        lambda ex: process_single_qa(
            tokenizer, ex, max_length, max_label_len=max_label_len, is_train=is_train
        ),
        batched=False,
        num_proc=num_proc,
    )
    if is_train:
        keep = {"input_ids", "attention_mask", "labels"}
        cols_to_remove = [c for c in mapped.column_names if c not in keep]
        mapped = mapped.remove_columns(cols_to_remove)
    return mapped


def tokenise_dataset_dict(
    dsdict: DatasetDict,
    tokenizer: AutoTokenizer,
    max_length: int = 4096,
    max_label_len: int | None = 1024,
    num_proc: int | None = None,
) -> Tuple[DatasetDict, DatasetDict]:
    """Return (full_dict, minimal_dict) after tokenisation.

    *minimal_dict* removes large meta columns for eval/test like the old code.
    """
    full_dict: Dict[str, Dataset] = {}
    minimal_dict: Dict[str, Dataset] = {}
    for split, ds in dsdict.items():
        is_train = split == "train"
        tok = tokenise_split(ds, tokenizer, max_length, max_label_len, is_train, num_proc)
        full_dict[split] = tok
        # For non-train we keep full but create minimal too
        minimal_dict[split] = tok.remove_columns([c for c in ds.column_names if c not in {"labels", "input_ids", "attention_mask"}])
    return DatasetDict(full_dict), DatasetDict(minimal_dict)
