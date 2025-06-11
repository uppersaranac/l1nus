#!/usr/bin/env python
"""CLI: fine-tune a causal-LM on tokenised dataset produced by cli_build.

This is a trimmed-down version of the original *train_llm.py* workflow that
assumes tokenisation/expansion/splitting are already done.  It keeps the same
metric and generation helpers via `llm_apis` to ensure regression parity.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from datasets import load_from_disk, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from llm_apis import compute_metrics_closure, do_generation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune a causal-LM on prepared dataset")
    p.add_argument("--dataset_dir", required=True, help="Directory with 'full/' & 'minimal/' sub-dirs")
    p.add_argument("--model_name", required=True, help="HF model checkpoint to fine-tune")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--logging_steps", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=1000)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--model_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    ds_full_path = Path(args.dataset_dir) / "full"
    ds_min_path = Path(args.dataset_dir) / "minimal"

    logger.info("Loading datasets from %s", ds_full_path)
    ds_full = load_from_disk(str(ds_full_path))
    ds_min = load_from_disk(str(ds_min_path))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=getattr(torch, args.model_dtype))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    targs = Seq2SeqTrainingArguments(
        output_dir=str(Path(args.output_dir).expanduser()),
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        predict_with_generate=True,
        generation_max_length=args.max_new_tokens,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        logging_steps=args.logging_steps,
        save_total_limit=2,
        metric_for_best_model="exact_match",
        bf16=args.model_dtype == "bfloat16",
        fp16=args.model_dtype == "float16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    compute_metrics = compute_metrics_closure(tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=ds_full["train"],
        eval_dataset=ds_min["valid"],
        compute_metrics=compute_metrics,
    )

    logger.info("Starting fine-tuning…")
    trainer.train()

    # quick generation sanity check
    logger.info("Running generation on 3 validation examples…")
    preds = do_generation(args.max_new_tokens, tokenizer, model, ds_min["valid"].select(range(3)))
    for p in preds:
        logger.info("PRED: %s", p)

    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
