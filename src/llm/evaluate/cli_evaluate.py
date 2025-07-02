#!/usr/bin/env python
"""
CLI: Evaluate a trained causal-LM on a tokenised dataset and output metrics and predictions.
"""
from __future__ import annotations

import argparse
import logging
import csv
from pathlib import Path

from accelerate import Accelerator
from datasets import load_from_disk
from llm.llm_apis import compute_metrics_closure, do_evaluate, do_generation
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a causal-LM on a prepared dataset")
    parser.add_argument("--dataset_dir", required=True, help="Directory with the dataset to evaluate")
    parser.add_argument("--model_name", required=True, help="HF model checkpoint to evaluate")
    parser.add_argument("--split", default="test", choices=["test", "valid", "val"], help="Dataset split to evaluate")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2, help="Batch size for evaluation")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens to generate")
    parser.add_argument("--limit", type=int, default=None, help="If set, truncate the evaluation set to this many examples")
    parser.add_argument("--output_csv", type=str, default=None, help="Path to CSV file for gold/prediction output (default: dataset_dir/eval_predictions.csv)")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    args.dataset_dir = str(Path(args.dataset_dir).expanduser())
    if args.output_csv is not None:
        args.output_csv = str(Path(args.output_csv).expanduser())
    accelerator = Accelerator()

    # Load dataset
    ds = load_from_disk(args.dataset_dir)
    split = args.split
    if split == "val":
        split = "valid"
    if split not in ds:
        raise ValueError(f"Split '{split}' not found in dataset at {args.dataset_dir}")
    dataset = ds[split]
    # Keep only the required columns for evaluation so that default_data_collator works
    columns_to_keep = ["input_ids", "attention_mask", "labels"]
    columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
    dataset = dataset.remove_columns(columns_to_remove)
    if args.limit is not None:
        dataset = dataset.shuffle(seed=42).select(range(min(args.limit, len(dataset))))
    logger.info(f"Loaded {len(dataset)} examples from split '{split}'")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model = accelerator.prepare(model)

    # default_data_collator casts data to pytorch tensors
    dataloader = DataLoader(dataset, batch_size=args.per_device_eval_batch_size, shuffle=False, collate_fn=default_data_collator)
    # prepare dataloader for distributed training, including moving data to device
    dataloader = accelerator.prepare(dataloader)

    compute_metrics = compute_metrics_closure(tokenizer)
    metrics = do_evaluate(
        accelerator,
        model,
        dataloader,
        tokenizer,
        compute_metrics,
        args.max_new_tokens,
        num_examples=len(dataset)
    )
    logger.info(f"Evaluation metrics: {metrics}")

    # Output CSV with gold labels and predictions
    if args.output_csv:
        csv_path = args.output_csv
    else:
        csv_path = str(Path(args.dataset_dir) / "eval_predictions.csv")
    logger.info(f"Generating predictions and writing to {csv_path}")
    preds = do_generation(args.max_new_tokens, tokenizer, accelerator.unwrap_model(model).eval(), dataset)
    dataset.set_format(type="torch", columns=["labels"])
    labels_tensor = dataset["labels"].masked_fill(dataset["labels"] == -100, tokenizer.pad_token_id)
    gold = tokenizer.batch_decode(labels_tensor, skip_special_tokens=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "gold_label", "prediction"])
        for i, (g, p) in enumerate(zip(gold, preds)):
            writer.writerow([i, g, p])
    logger.info("CSV file written.")

if __name__ == "__main__":
    main()
