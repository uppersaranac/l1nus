#!/usr/bin/env python
"""CLI: fine-tune a causal-LM on tokenised dataset produced by cli_build.

This is a trimmed-down version of the original *train_llm.py* workflow that
assumes tokenisation/expansion/splitting are already done.  It keeps the same
metric and generation helpers via `llm_apis` to ensure regression parity.
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import cast

from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import DatasetDict, load_from_disk
from llm.llm_apis import compute_metrics_closure, do_evaluate
from torch.utils.data import DataLoader  # local import to avoid circular issues
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.data.data_collator import default_data_collator
from transformers.optimization import get_scheduler
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for fine-tuning a causal-LM on a prepared dataset.

    :return: Parsed arguments namespace.
    :rtype: argparse.Namespace
    """
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
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--num_example_preds", type=int, default=3, help="Number of example predictions to log during evaluation")
    p.add_argument("--eval_num_examples", type=int, default=100, help="Number of examples to use for metric computation during evaluation")
    p.add_argument("--model_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--no_tqdm", action="store_true", help="Disable tqdm progress bars.")
    p.add_argument("--limit", type=int, default=None, help="If set, truncate the training set to this many examples.")
    p.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty for generation")
    p.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    p.add_argument("--do_sample", action="store_true", help="Use sampling for generation")
    p.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling threshold")
    p.add_argument("--lr_scheduler_type", type=str, default="cosine", 
                   choices=["linear", "linear_with_warmup", "cosine"], 
                   help="Learning rate scheduler type (default: cosine)")
    p.add_argument("--warmup_ratio", type=float, default=0.1, 
                   help="Warmup ratio (fraction of total steps) for warmup schedulers (default: 0.1)")
    p.add_argument("--use_position_weighting", action="store_true", 
                   help="Enable position-wise loss weighting from dataset")
    p.add_argument("--weight_column", type=str, default="position_weights", 
                   help="Name of the column containing position weights in the dataset (default: position_weights)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------



def main() -> None:
    args = _parse_args()

    # Initialise Accelerator (handles DDP, fp16/bf16, etc.)
    ddp_handler = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_handler],
    )

    # Ensure only rank-0 logs verbosely
    if not accelerator.is_main_process:
        logger.setLevel(logging.ERROR)

    # ---------------- data ----------------
    dataset_dir = Path(args.dataset_dir).expanduser()
    ds_min_path = dataset_dir / "minimal"
    ds_min = cast(DatasetDict, load_from_disk(str(ds_min_path)))
    # Add a type check to ensure we have a DatasetDict
    if not isinstance(ds_min, DatasetDict):
        raise TypeError(f"Expected a DatasetDict from {ds_min_path}, but got {type(ds_min)}")

    train_dataset = ds_min["train"]
    if args.limit is not None and args.limit > 0:
        train_dataset = train_dataset.select(range(min(args.limit, len(train_dataset))))
    eval_dataset = ds_min["valid"].shuffle(seed=42).select(range(min(args.eval_num_examples, len(ds_min["valid"]))))

    # ---------------- model & optim ----------------
    model_name = Path(args.model_name).expanduser()
    tokenizer = AutoTokenizer.from_pretrained(str(model_name))
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_name),
        torch_dtype=getattr(torch, args.model_dtype),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # ---------------- dataloaders ----------------

    # Configure dataset format and columns based on whether position weighting is enabled
    if args.use_position_weighting:
        # Verify that the weight column exists in the dataset
        if args.weight_column not in train_dataset.column_names:
            raise ValueError(f"Weight column '{args.weight_column}' not found in dataset. "
                           f"Available columns: {train_dataset.column_names}")
        
        # Set format to include weight column along with standard columns
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", args.weight_column])
    else:
        # Standard format with only the required columns
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_loader = DataLoader(
        train_dataset,  # type: ignore[arg-type]
        shuffle=True,
        batch_size=args.per_device_train_batch_size,
        collate_fn=default_data_collator,
    )
    eval_loader = DataLoader(
        eval_dataset,  # type: ignore[arg-type]
        shuffle=False,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=default_data_collator,
    )

    # Prepare everything for distributed/accelerated execution
    model, optimizer, train_loader, eval_loader = accelerator.prepare(
        model, optimizer, train_loader, eval_loader
    )

    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    num_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Configure learning rate scheduler based on command line option
    if args.lr_scheduler_type == "linear":
        # Original linear scheduler with no warmup
        num_warmup_steps = 0
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )
    elif args.lr_scheduler_type == "linear_with_warmup":
        # Linear scheduler with warmup
        num_warmup_steps = int(args.warmup_ratio * num_train_steps)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )
    elif args.lr_scheduler_type == "cosine":
        # Cosine annealing with warmup (default)
        num_warmup_steps = int(args.warmup_ratio * num_train_steps)
        lr_scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {args.lr_scheduler_type}")

    if accelerator.is_main_process:
        logger.info("Using %s scheduler with %d warmup steps out of %d total steps", 
                   args.lr_scheduler_type, 
                   num_warmup_steps, 
                   num_train_steps)
        if args.use_position_weighting:
            logger.info("Using position-wise loss weighting from dataset column: %s", args.weight_column)
        else:
            logger.info("Using standard uniform loss weighting")

    # Metric helper (exact-match) ------------------------------------
    compute_metrics = compute_metrics_closure(tokenizer)

    # ---------------- training loop ----------------
    global_step = 0
    best_exact = -1.0
    for epoch in (tqdm(range(args.num_train_epochs), desc="Epoch", disable=args.no_tqdm) if not args.no_tqdm else range(args.num_train_epochs)):
        model.train()
        train_iter = train_loader
        if not args.no_tqdm:
            train_iter = tqdm(train_loader, desc=f"Train (epoch {epoch+1})", leave=False)
        last_loss = None
        for batch in train_iter:
            with accelerator.accumulate(model):
                if args.use_position_weighting and args.weight_column in batch:
                    # Use position weights from dataset
                    position_weights = batch[args.weight_column]
                    
                    # Get logits without computing loss
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    
                    # Compute weighted loss manually
                    logits = outputs.logits
                    labels = batch["labels"]
                    
                    # Shift logits and labels for causal LM
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    shift_weights = position_weights[..., 1:].contiguous()
                    
                    # Flatten for loss computation
                    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                    shift_labels = shift_labels.view(-1)
                    shift_weights = shift_weights.view(-1)
                    
                    # Compute cross entropy loss without reduction
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    losses = loss_fct(shift_logits, shift_labels)
                    
                    # Apply position weights and reduce
                    weighted_losses = losses * shift_weights
                    # Only average over non-masked tokens
                    valid_tokens = (shift_labels != -100)
                    if valid_tokens.sum() > 0:
                        loss = weighted_losses[valid_tokens].mean()
                    else:
                        loss = weighted_losses.mean()
                else:
                    # Standard loss computation
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    loss = outputs.loss
                
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            last_loss = loss.item()
            if not args.no_tqdm and hasattr(train_iter, 'set_postfix'):
                train_iter.set_postfix(loss=f"{last_loss:.4f}")
            else:
                # ------------ logging & eval -------------
                if global_step % args.logging_steps == 0 and accelerator.is_main_process:
                    logger.info("Epoch %d | step %d | loss %.4f", epoch, global_step, last_loss)

            if global_step % args.eval_steps == 0 and global_step != 0:
                eval_metrics = do_evaluate(
                    accelerator, model, eval_loader, tokenizer, compute_metrics, 
                    args.max_new_tokens, args.num_example_preds,
                    repetition_penalty=args.repetition_penalty,
                    temperature=args.temperature,
                    do_sample=args.do_sample,
                    top_p=args.top_p
                )
                if not args.no_tqdm and hasattr(train_iter, 'set_postfix') and eval_metrics is not None:
                    metric_val = eval_metrics.get('exact_match', None)
                    if metric_val is not None:
                        train_iter.set_postfix(loss=f"{last_loss:.4f}", exact_match=f"{metric_val:.4f}")

            global_step += 1

        # ----- end-of-epoch evaluation & best-model saving -----
        epoch_metrics = do_evaluate(
            accelerator,
            model,
            eval_loader,
            tokenizer,
            compute_metrics,
            args.max_new_tokens,
            args.eval_num_examples,
            repetition_penalty=args.repetition_penalty,
            temperature=args.temperature,
            do_sample=args.do_sample,
            top_p=args.top_p
        )
        if not args.no_tqdm and 'train_iter' in locals() and hasattr(train_iter, 'set_postfix') and epoch_metrics is not None:
            metric_val = epoch_metrics.get('exact_match', None)
            if metric_val is not None:
                train_iter.set_postfix(exact_match=f"{metric_val:.4f}")
        if accelerator.is_main_process:
            exact_val = epoch_metrics.get("exact_match", 0)
            if exact_val >= best_exact:
                best_exact = exact_val
                best_dir = Path(args.output_dir).expanduser() / "best_model"
                accelerator.print(f"New best exact_match {exact_val:.4f} → saving model to {best_dir}")
                accelerator.unwrap_model(model).save_pretrained(
                    best_dir,
                    save_function=accelerator.save,
                    safe_serialization=True,
                )
                tokenizer.save_pretrained(best_dir)

    # Final evaluation and save ---------------------------------------
    epoch_metrics = do_evaluate(
        accelerator,
        model,
        eval_loader,
        tokenizer,
        compute_metrics,
        args.max_new_tokens,
        args.eval_num_examples,
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        do_sample=args.do_sample,
        top_p=args.top_p
    )
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        out_dir = Path(args.output_dir).expanduser()
        unwrapped.save_pretrained(out_dir, save_function=accelerator.save, safe_serialization=True)
        tokenizer.save_pretrained(out_dir)
        logger.info("Model and tokenizer saved to %s", out_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
