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
    default_data_collator,
    get_scheduler,
)
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm import tqdm


from llm.llm_apis import compute_metrics_closure, do_generation

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
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--num_example_preds", type=int, default=3, help="Number of example predictions to log during evaluation")
    p.add_argument("--eval_num_examples", type=int, default=100, help="Number of examples to use for metric computation during evaluation")
    p.add_argument("--model_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--no_tqdm", action="store_true", help="Disable tqdm progress bars.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _evaluate(accelerator: Accelerator, model, dataloader, tokenizer, compute_metrics, max_new_tokens: int, num_examples: int = 100) -> dict:
    """Run generation-based evaluation and log exact-match metric."""
    model.eval()
    # Only evaluate over the first num_examples examples in the dataloader
    num_processed = 0
    for batch in dataloader:
        batch_size = batch["input_ids"].size(0)
        # If adding this batch would exceed num_examples, trim the batch
        if num_processed + batch_size > num_examples:
            trim = num_examples - num_processed
            for k in batch:
                batch[k] = batch[k][:trim]
            batch_size = trim
        with torch.no_grad():
            generated = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
            )
        # Gather across processes so that metric is computed on full set
        gen_all = accelerator.gather(generated)
        labels_all = accelerator.gather(batch["labels"])
        # Stream into metric accumulator (compute_result=False)
        compute_metrics((gen_all.cpu().numpy(), labels_all.cpu().numpy()), compute_result=False)
        num_processed += batch_size
        if num_processed >= num_examples:
            break
    # Finalise metric
    metrics = compute_metrics((torch.empty(0), torch.empty(0)), compute_result=True)
    if accelerator.is_main_process:
        logger.info("Evaluation metrics: %s", metrics)

        # ---- log example predictions ----
        try:
            sample_ds = dataloader.dataset.select(range(min(num_examples, len(dataloader.dataset))))
            preds = do_generation(
                max_new_tokens,
                tokenizer,
                accelerator.unwrap_model(model).eval(),
                sample_ds,
            )
            # decode gold labels
            sample_ds.set_format(type="torch", columns=["labels"])
            labels_tensor = sample_ds["labels"].masked_fill(sample_ds["labels"] == -100, tokenizer.pad_token_id)
            gold = tokenizer.batch_decode(labels_tensor, skip_special_tokens=True)
            for i, (g_pred, g_gold) in enumerate(zip(preds, gold)):
                logger.info("EXAMPLE %d | PRED: %s | GOLD: %s", i, g_pred, g_gold)
        except Exception as e:
            logger.warning("Failed to generate example predictions: %s", e)

    model.train()
    return metrics


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
    ds_full_path = Path(args.dataset_dir) / "full"
    ds_min_path = Path(args.dataset_dir) / "minimal"
    if accelerator.is_main_process:
        logger.info("Loading tokenised datasets from %s", ds_full_path)
    ds_full = load_from_disk(str(ds_full_path))
    ds_min = load_from_disk(str(ds_min_path))

    train_dataset = ds_full["train"]
    eval_dataset = ds_min["valid"]

    # ---------------- model & optim ----------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=getattr(torch, args.model_dtype),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # ---------------- dataloaders ----------------
    from torch.utils.data import DataLoader  # local import to avoid circular issues
    import math

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.per_device_train_batch_size,
        collate_fn=default_data_collator,
    )
    eval_loader = DataLoader(
        eval_dataset,
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

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps,
    )

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

            # ------------ logging & eval -------------
            if global_step % args.logging_steps == 0 and accelerator.is_main_process:
                logger.info("Epoch %d | step %d | loss %.4f", epoch, global_step, last_loss)

            if global_step % args.eval_steps == 0 and global_step != 0:
                eval_metrics = _evaluate(accelerator, model, eval_loader, tokenizer, compute_metrics, args.max_new_tokens, args.num_example_preds)
                if not args.no_tqdm and hasattr(train_iter, 'set_postfix') and eval_metrics is not None:
                    metric_val = eval_metrics.get('exact_match', None)
                    if metric_val is not None:
                        train_iter.set_postfix(loss=f"{last_loss:.4f}", exact_match=f"{metric_val:.4f}")

            global_step += 1

        # ----- end-of-epoch evaluation & best-model saving -----
        epoch_metrics = _evaluate(
            accelerator,
            model,
            eval_loader,
            tokenizer,
            compute_metrics,
            args.max_new_tokens,
            args.eval_num_examples,
        )
        if not args.no_tqdm and 'train_iter' in locals() and hasattr(train_iter, 'set_postfix') and epoch_metrics is not None:
            metric_val = epoch_metrics.get('exact_match', None)
            if metric_val is not None:
                train_iter.set_postfix(exact_match=f"{metric_val:.4f}")
        if accelerator.is_main_process:
            exact_val = epoch_metrics.get("exact_match", 0)
            if exact_val > best_exact:
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
    _evaluate(accelerator, model, eval_loader, tokenizer, compute_metrics, args.max_new_tokens, args.num_example_preds)

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        out_dir = Path(args.output_dir).expanduser()
        unwrapped.save_pretrained(out_dir, save_function=accelerator.save, safe_serialization=True)
        tokenizer.save_pretrained(out_dir)
        logger.info("Model and tokenizer saved to %s", out_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
