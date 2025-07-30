#!/usr/bin/env python
"""
CLI: Evaluate a trained causal-LM on a tokenised dataset and output metrics and predictions.
"""

import argparse
import logging
import csv
from pathlib import Path

from accelerate import Accelerator
from datasets import DatasetDict, load_from_disk
from llm.llm_apis import compute_metrics_closure, do_evaluate, do_generation, _norm_tagged
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.data.data_collator import default_data_collator
from collections import defaultdict

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
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty for generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling for generation")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling threshold")
    return parser.parse_args()

def analyze_predictions_by_question_id(question_ids: list, gold_labels: list[str], predictions: list[str], tokenizer, show_examples: bool = False) -> None:
    """
    Analyze predictions by question_id, breaking down by whether gold answer is 0 or not.
    
    :param question_ids: List of question IDs for each example
    :param gold_labels: List of gold label strings
    :param predictions: List of prediction strings 
    :param tokenizer: Tokenizer instance (for compatibility with _norm_tagged)
    :param show_examples: Whether to show individual examples (default: False)
    """
    # Group data by question_id
    question_data = defaultdict(list)
    
    for qid, gold, pred in zip(question_ids, gold_labels, predictions):
        question_data[qid].append({
            'gold': gold,
            'pred': pred
        })
    
    # Analyze each question_id
    zero_answer_stats = {'total': 0, 'correct': 0}
    non_zero_answer_stats = {'total': 0, 'correct': 0}
    
    logger.info(f"Found {len(question_data)} unique question_ids")
    
    for qid, examples in question_data.items():
        qid_correct = 0
        qid_total = len(examples)
        qid_zero_correct = 0
        qid_zero_total = 0
        qid_non_zero_correct = 0
        qid_non_zero_total = 0
        
        logger.info(f"\nQuestion ID: {qid}")
        
        # Extract normalized answers using _norm_tagged
        for i, example in enumerate(examples):
            gold_answer = _norm_tagged(example['gold'], tokenizer)
            pred_answer = _norm_tagged(example['pred'], tokenizer)
            
            is_correct = gold_answer == pred_answer
            qid_correct += is_correct
            
            # Categorize by whether gold answer is 0 or not
            if gold_answer == "0":
                zero_answer_stats['total'] += 1
                qid_zero_total += 1
                if is_correct:
                    zero_answer_stats['correct'] += 1
                    qid_zero_correct += 1
            else:
                non_zero_answer_stats['total'] += 1
                qid_non_zero_total += 1
                if is_correct:
                    non_zero_answer_stats['correct'] += 1
                    qid_non_zero_correct += 1
            
            if show_examples:
                logger.info(f"  Example {i+1}:")
                logger.info(f"    Gold: '{gold_answer}'")
                logger.info(f"    Pred: '{pred_answer}'")
                logger.info(f"    Correct: {is_correct}")
        
        # Print statistics for this question ID
        qid_accuracy = qid_correct / qid_total if qid_total > 0 else 0
        logger.info(f"  Total examples: {qid_total}")
        logger.info(f"  Correct: {qid_correct}")
        logger.info(f"  Accuracy: {qid_accuracy:.3f}")
        
        if qid_zero_total > 0:
            qid_zero_accuracy = qid_zero_correct / qid_zero_total
            logger.info(f"  Zero answers: {qid_zero_correct}/{qid_zero_total} ({qid_zero_accuracy:.3f})")
        
        if qid_non_zero_total > 0:
            qid_non_zero_accuracy = qid_non_zero_correct / qid_non_zero_total
            logger.info(f"  Non-zero answers: {qid_non_zero_correct}/{qid_non_zero_total} ({qid_non_zero_accuracy:.3f})")
    
    # Print summary statistics
    logger.info("\n" + "="*50)
    logger.info("SUMMARY STATISTICS BY ANSWER TYPE")
    logger.info("="*50)
    
    if zero_answer_stats['total'] > 0:
        zero_accuracy = zero_answer_stats['correct'] / zero_answer_stats['total']
        logger.info("Gold answer = '0':")
        logger.info(f"  Total examples: {zero_answer_stats['total']}")
        logger.info(f"  Correct predictions: {zero_answer_stats['correct']}")
        logger.info(f"  Accuracy: {zero_accuracy:.3f}")
    else:
        logger.info("Gold answer = '0': No examples found")
    
    if non_zero_answer_stats['total'] > 0:
        non_zero_accuracy = non_zero_answer_stats['correct'] / non_zero_answer_stats['total']
        logger.info("Gold answer != '0':")
        logger.info(f"  Total examples: {non_zero_answer_stats['total']}")
        logger.info(f"  Correct predictions: {non_zero_answer_stats['correct']}")
        logger.info(f"  Accuracy: {non_zero_accuracy:.3f}")
    else:
        logger.info("Gold answer != '0': No examples found")
    
    total_examples = zero_answer_stats['total'] + non_zero_answer_stats['total']
    total_correct = zero_answer_stats['correct'] + non_zero_answer_stats['correct']
    if total_examples > 0:
        overall_accuracy = total_correct / total_examples
        logger.info("Overall:")
        logger.info(f"  Total examples: {total_examples}")
        logger.info(f"  Correct predictions: {total_correct}")
        logger.info(f"  Accuracy: {overall_accuracy:.3f}")

def main() -> None:
    args = parse_args()
    args.dataset_dir = str(Path(args.dataset_dir).expanduser())
    if args.output_csv is not None:
        args.output_csv = str(Path(args.output_csv).expanduser())
    accelerator = Accelerator()

    # Load dataset
    ds = load_from_disk(args.dataset_dir)
    # Add a type check to ensure we have a DatasetDict
    if not isinstance(ds, DatasetDict):
        raise TypeError(f"Expected a DatasetDict from {args.dataset_dir}, but got {type(ds)}")
    
    split = args.split
    if split == "val":
        split = "valid"
    if split not in ds:
        raise ValueError(f"Split '{split}' not found in dataset at {args.dataset_dir}")
    dataset = ds[split]
    
    # Store the original question_id for analysis
    original_question_ids = dataset["question_id"] if "question_id" in dataset.column_names else None
    
    # Keep only the required columns for evaluation so that default_data_collator works
    columns_to_keep = ["input_ids", "attention_mask", "labels"]
    columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
    dataset = dataset.remove_columns(columns_to_remove)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
        # Also limit the question_ids if they exist
        if original_question_ids is not None:
            original_question_ids = original_question_ids[:min(args.limit, len(original_question_ids))]
    logger.info(f"Loaded {len(dataset)} examples from split '{split}'")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model = accelerator.prepare(model)

    # default_data_collator casts data to pytorch tensors
    dataloader = DataLoader(dataset,   # type: ignore[arg-type]
                            batch_size=args.per_device_eval_batch_size, 
                            shuffle=False, 
                            collate_fn=default_data_collator)
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
        num_examples=len(dataset),
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        do_sample=args.do_sample,
        top_p=args.top_p
    )
    logger.info(f"Evaluation metrics: {metrics}")

    # Output CSV with gold labels and predictions
    if args.output_csv:
        csv_path = args.output_csv
    else:
        csv_path = str(Path(args.dataset_dir) / "eval_predictions.csv")
    logger.info(f"Generating predictions and writing to {csv_path}")
    preds = do_generation(
        args.max_new_tokens,
        tokenizer,
        accelerator.unwrap_model(model).eval(),
        dataset,
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        do_sample=args.do_sample,
        top_p=args.top_p
    )
    dataset.set_format(type="torch", columns=["labels"])
    labels_tensor = dataset["labels"].masked_fill(dataset["labels"] == -100, tokenizer.pad_token_id) # type: ignore[arg-type]
    gold = tokenizer.batch_decode(labels_tensor, skip_special_tokens=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "gold_label", "prediction"])
        for i, (g, p) in enumerate(zip(gold, preds)):
            writer.writerow([i, g, p])
    logger.info("CSV file written.")
    
    # Analyze predictions by question_id if available
    if original_question_ids is not None:
        logger.info("Analyzing predictions by question_id...")
        analyze_predictions_by_question_id(original_question_ids, gold, preds, tokenizer, show_examples=False)
    else:
        logger.warning("question_id column not found in dataset - skipping question_id analysis")

if __name__ == "__main__":
    main()
