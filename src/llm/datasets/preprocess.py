"""Utilities to convert *questions.jsonl* into tokenised HuggingFace Datasets.

This module deliberately depends on the *existing* `llm_apis.process_single_qa`
function so that tokenisation behaviour is identical to the original pipeline
— this is crucial for regression-test parity.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple, cast

import datasets as hfds
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from llm.llm_apis import process_single_qa

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading & splitting helpers
# ---------------------------------------------------------------------------

def load_questions_jsonl(path: str | Path) -> Dataset:
    """Load the JSONL file produced by *cli_generate.py* into a HF Dataset."""
    # the split argument is set to load all of the data (train, valid, test) into a flat Dataset.
    ds = cast(Dataset, hfds.load_dataset("json", data_files=str(path), split="train")) # , streaming=True)  streaming dataset doesn't have column names!
    logger.info("Loaded Q-A records from %s", path)
    return ds


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
    system_prompt: str | None = None,
    create_position_weights: bool = False,
    default_weight: float = 1.0,
    answer_weight: float = 2.0,
) -> Dataset:
    """
    Apply *process_single_qa* to every example in *ds*.
    
    :param ds: Dataset to tokenize
    :type ds: Dataset
    :param tokenizer: Tokenizer instance
    :type tokenizer: AutoTokenizer
    :param max_length: Maximum sequence length
    :type max_length: int
    :param max_label_len: Maximum label length for evaluation
    :type max_label_len: int or None
    :param is_train: Whether this is training data
    :type is_train: bool
    :param num_proc: Number of processes for parallel processing
    :type num_proc: int or None
    :param system_prompt: System prompt override
    :type system_prompt: str or None
    :param create_position_weights: Whether to create position weights based on answer tags
    :type create_position_weights: bool
    :param default_weight: Weight for positions outside answer tags
    :type default_weight: float
    :param answer_weight: Weight for positions inside answer tags
    :type answer_weight: float
    :return: Tokenized dataset
    :rtype: Dataset
    """
    logger.info("Tokenising %d examples (is_train=%s, create_position_weights=%s)…", len(ds), is_train, create_position_weights)
    mapped = ds.map(
        lambda ex: process_single_qa(
            tokenizer,
            ex,
            max_length,
            max_label_len=max_label_len,
            is_train=is_train,
            system_prompt_override=system_prompt,
            create_position_weights=create_position_weights and is_train,  # Only create weights for training data
            default_weight=default_weight,
            answer_weight=answer_weight,
        ),
        batched=False,
        num_proc=num_proc,
    )
    if is_train:
        # Keep position_weights column if it was created
        keep = {"input_ids", "attention_mask", "labels"}
        if create_position_weights:
            keep.add("position_weights")
        cols_to_remove = [c for c in mapped.column_names if c not in keep]
        mapped = mapped.remove_columns(cols_to_remove)
    return mapped


def tokenise_dataset_dict(
    dsdict: DatasetDict,
    tokenizer: AutoTokenizer,
    max_length: int = 4096,
    max_label_len: int | None = 1024,
    num_proc: int | None = None,
    system_prompt: str | None = None,
    create_position_weights: bool = False,
    default_weight: float = 1.0,
    answer_weight: float = 10.0,
) -> Tuple[DatasetDict, DatasetDict]:
    """
    Return (full_dict, minimal_dict) after tokenisation.

    *minimal_dict* removes large meta columns for eval/test like the old code.
    
    :param dsdict: Dictionary of datasets to tokenize
    :type dsdict: DatasetDict
    :param tokenizer: Tokenizer instance
    :type tokenizer: AutoTokenizer
    :param max_length: Maximum sequence length
    :type max_length: int
    :param max_label_len: Maximum label length for evaluation
    :type max_label_len: int or None
    :param num_proc: Number of processes for parallel processing
    :type num_proc: int or None
    :param system_prompt: System prompt override
    :type system_prompt: str or None
    :param create_position_weights: Whether to create position weights based on answer tags
    :type create_position_weights: bool
    :param default_weight: Weight for positions outside answer tags
    :type default_weight: float
    :param answer_weight: Weight for positions inside answer tags
    :type answer_weight: float
    :return: Tuple of (full_dict, minimal_dict) after tokenization
    :rtype: Tuple[DatasetDict, DatasetDict]
    """
    full_dict: Dict[str, Dataset] = {}
    minimal_dict: Dict[str, Dataset] = {}
    for split, ds in dsdict.items():
        is_train = split == "train"
        tok = tokenise_split(
            ds,
            tokenizer,
            max_length,
            max_label_len,
            is_train,
            num_proc,
            system_prompt=system_prompt,
            create_position_weights=create_position_weights,
            default_weight=default_weight,
            answer_weight=answer_weight,
        )

        if is_train:
            # Exclude the training set from the *full* dataset but keep a minimal version
            logger.info("Excluding 'train' split from full tokenised dataset as requested")
            minimal_dict[split] = tok  # already minimal due to is_train=True
        else:
            # Keep full information for validation / test splits
            full_dict[split] = tok
            # Create minimal version by stripping meta columns
            minimal_cols = {"labels", "input_ids", "attention_mask"}
            minimal_dict[split] = tok.remove_columns([
                c for c in tok.column_names if c not in minimal_cols
            ])

    return DatasetDict(full_dict), DatasetDict(minimal_dict)
