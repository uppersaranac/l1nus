#!/usr/bin/env python
"""CLI: examine and sample datasets created by cli_build.py.

Example:

    python -m llm.datasets.cli_sample \
        --dataset_dir data/chem_prop_ds \
        --model_name openai/gpt2 \
        --num_examples 5
"""

import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List
import pyarrow.compute as pc

from datasets import DatasetDict, load_from_disk
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for dataset sampling and analysis.

    :return: Parsed arguments namespace.
    :rtype: argparse.Namespace
    """
    p = argparse.ArgumentParser(description="Examine and sample datasets created by cli_build.py")
    p.add_argument("--dataset_dir", required=True, help="Directory containing 'full/' & 'minimal/' sub-dirs")
    p.add_argument("--model_name", required=True, help="HF tokenizer name/path to use for decoding")
    p.add_argument("--num_examples", type=int, default=5, help="Number of random examples to decode and display")
    return p.parse_args()


def analyze_padding_tokens(input_ids: List[List[int]], pad_token_id: int) -> Dict[str, float]:
    """
    Efficiently analyze padding tokens in input_ids column using pyarrow ChunkedArray.
    :param input_ids: pyarrow.ChunkedArray of tokenized sequences
    :param pad_token_id: The padding token ID
    :return: Dictionary with min, max, mean padding token counts
    """
    # input_ids: pyarrow.ChunkedArray of lists
    # Compare each value to pad_token_id
    eq_mask = pc.equal(input_ids, pad_token_id)
    # eq_mask: ChunkedArray of lists of bools
    pad_counts = pc.list_sum(eq_mask)
    min_val = pc.min(pad_counts).as_py() if pad_counts else 0
    max_val = pc.max(pad_counts).as_py() if pad_counts else 0
    mean_val = pc.mean(pad_counts).as_py() if pad_counts else 0.0
    return {
        "min": float(min_val),
        "max": float(max_val),
        "mean": float(mean_val)
    }


def analyze_label_masks(labels: List[List[int]]) -> Dict[str, float]:
    """
    Efficiently analyze -100 values in labels column using pyarrow ChunkedArray.
    :param labels: pyarrow.ChunkedArray of label sequences
    :return: Dictionary with min, max, mean -100 counts
    """
    eq_mask = pc.equal(labels, -100)
    mask_counts = pc.list_sum(eq_mask)
    min_val = pc.min(mask_counts).as_py() if mask_counts else 0
    max_val = pc.max(mask_counts).as_py() if mask_counts else 0
    mean_val = pc.mean(mask_counts).as_py() if mask_counts else 0.0
    return {
        "min": float(min_val),
        "max": float(max_val),
        "mean": float(mean_val)
    }


def decode_labels_skip_mask(tokenizer: AutoTokenizer, labels: List[int]) -> str:
    """
    Decode labels while skipping -100 values.

    :param tokenizer: Tokenizer instance
    :param labels: Label sequence with -100 mask values
    :return: Decoded string
    """
    # Filter out -100 values
    filtered_labels = [token_id for token_id in labels if token_id != -100]
    
    if not filtered_labels:
        return "[No valid labels]"
    
    try:
        return tokenizer.decode(filtered_labels, skip_special_tokens=True)
    except Exception as e:
        logger.warning("Failed to decode labels: %s", e)
        return f"[Decode error: {e}]"


def sample_and_decode_examples(dataset, tokenizer: AutoTokenizer, num_examples: int, dataset_name: str) -> None:
    """
    Sample random examples from dataset and decode them.

    :param dataset: Dataset to sample from
    :param tokenizer: Tokenizer for decoding
    :param num_examples: Number of examples to sample
    :param dataset_name: Name of the dataset for logging
    """
    dataset_size = len(dataset)
    if dataset_size == 0:
        logger.info("Dataset %s is empty, skipping sampling", dataset_name)
        return
    
    # Sample random indices
    num_to_sample = min(num_examples, dataset_size)
    rng = np.random.default_rng(seed=42)
    indices = rng.choice(dataset_size, size=num_to_sample, replace=False)
    indices.sort()
    
    logger.info("\n" + "="*80)
    logger.info("RANDOM SAMPLES FROM %s (%d examples)", dataset_name.upper(), num_to_sample)
    logger.info("="*80)
    
    for i, idx in enumerate(indices):
        example = dataset[int(idx)]
        
        logger.info(f"\n--- Example {i+1}/{num_to_sample} (index {idx}) ---")
        
        # Decode input_ids
        if "input_ids" in example:
            input_text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
            logger.info("INPUT: %s", input_text)
        
        # Decode labels (skipping -100)
        if "labels" in example:
            label_text = decode_labels_skip_mask(tokenizer, example["labels"])
            logger.info("LABELS: %s", label_text)
        
        logger.info("-" * 40)


def analyze_dataset(dataset, tokenizer: AutoTokenizer, dataset_name: str) -> None:
    """
    Analyze a single dataset for padding and label statistics.

    :param dataset: Dataset to analyze
    :param tokenizer: Tokenizer instance
    :param dataset_name: Name of the dataset
    """
    logger.info("\nAnalyzing dataset: %s", dataset_name)
    logger.info("Dataset size: %d", len(dataset))
    logger.info("Available columns: %s", dataset.column_names)
    
    # Analyze input_ids for padding
    if "input_ids" in dataset.column_names:
        input_ids_arr = dataset.data.column("input_ids")
        padding_stats = analyze_padding_tokens(input_ids_arr, tokenizer.pad_token_id)
        logger.info("Padding token statistics (pad_token_id=%d):", tokenizer.pad_token_id)
        logger.info("  Min padding tokens per row: %.1f", padding_stats["min"])
        logger.info("  Max padding tokens per row: %.1f", padding_stats["max"])
        logger.info("  Mean padding tokens per row: %.1f", padding_stats["mean"])
    
    # Analyze labels for -100 values
    if "labels" in dataset.column_names:
        labels_arr = dataset.data.column("labels")
        mask_stats = analyze_label_masks(labels_arr)
        logger.info("Label mask statistics (-100 values):")
        logger.info("  Min -100 values per row: %.1f", mask_stats["min"])
        logger.info("  Max -100 values per row: %.1f", mask_stats["max"])
        logger.info("  Mean -100 values per row: %.1f", mask_stats["mean"])


def analyze_dataset_dict(dataset_dict: DatasetDict, tokenizer: AutoTokenizer, dict_name: str, num_examples: int) -> None:
    """
    Analyze all datasets in a DatasetDict.

    :param dataset_dict: DatasetDict containing multiple splits
    :param tokenizer: Tokenizer instance
    :param dict_name: Name of the DatasetDict (e.g., "minimal", "full")
    :param num_examples: Number of examples to sample for decoding
    """
    logger.info("\n" + "="*60)
    logger.info("ANALYZING %s DATASET DICT", dict_name.upper())
    logger.info("="*60)
    
    for split_name, dataset in dataset_dict.items():
        # Remove all columns except input_ids and labels to minimize memory usage
        keep_cols = [col for col in dataset.column_names if col in ("input_ids", "labels")]
        drop_cols = [col for col in dataset.column_names if col not in keep_cols]
        if drop_cols:
            dataset = dataset.remove_columns(drop_cols)
        analyze_dataset(dataset, tokenizer, f"{dict_name}/{split_name}")
        # Sample and decode examples for each split
        sample_and_decode_examples(dataset, tokenizer, num_examples, f"{dict_name}/{split_name}")


def main() -> None:
    """Main function to analyze datasets."""
    args = _parse_args()
    
    dataset_dir = Path(args.dataset_dir).expanduser()
    
    # Load tokenizer
    logger.info("Loading tokenizer: %s", args.model_name)
    model_name = Path(args.model_name).expanduser()
    tokenizer = AutoTokenizer.from_pretrained(str(model_name))
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Pad token ID: %d", tokenizer.pad_token_id)
    logger.info("EOS token ID: %d", tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1)
    
    # Check for minimal and full dataset directories
    minimal_path = dataset_dir / "minimal"
    full_path = dataset_dir / "full"
    
    if minimal_path.exists():
        logger.info("Loading minimal dataset from: %s", minimal_path)
        try:
            minimal_ds = load_from_disk(str(minimal_path))
            if isinstance(minimal_ds, DatasetDict):
                analyze_dataset_dict(minimal_ds, tokenizer, "minimal", args.num_examples)
            else:
                logger.warning("Expected DatasetDict but got %s for minimal dataset", type(minimal_ds))
        except Exception as e:
            logger.error("Failed to load minimal dataset: %s", e)
    else:
        logger.warning("Minimal dataset directory not found: %s", minimal_path)
    
    if full_path.exists():
        logger.info("Loading full dataset from: %s", full_path)
        try:
            full_ds = load_from_disk(str(full_path))
            if isinstance(full_ds, DatasetDict):
                analyze_dataset_dict(full_ds, tokenizer, "full", args.num_examples)
            else:
                logger.warning("Expected DatasetDict but got %s for full dataset", type(full_ds))
        except Exception as e:
            logger.error("Failed to load full dataset: %s", e)
    else:
        logger.warning("Full dataset directory not found: %s", full_path)


if __name__ == "__main__":
    main()
