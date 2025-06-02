#!/usr/bin/env python
from __future__ import annotations
import argparse, logging, gc, sys
from pathlib import Path
import os

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    Seq2SeqTrainingArguments,
)
from accelerate import Accelerator


accelerator = Accelerator()

from llm.llm_apis import (
    load_arrow_dataset,
    process_single_qa,
    compute_metrics_closure,
    show_examples,
    do_generation,
    QUESTION_SETS,
    PROCESSOR_CLASSES,
)
from typing import Any, Sequence, Dict

"""
property_schema_fields = [
    pa.field("cid", pa.uint64()),
    pa.field("complexity", pa.float32()),
    pa.field("hba", pa.int32()),
    pa.field("hbd", pa.int32()),
    pa.field("rotatable_bonds", pa.int32()),
    pa.field("tpsa", pa.float32()),
    pa.field("logp", pa.float32()),
    pa.field("monoisotopic_mass", pa.float64()),
    pa.field("exact_mass", pa.float64()),
    pa.field("formula", pa.string()),
    pa.field("molecular_weight", pa.float64()),
    pa.field("charge", pa.int32()),
    pa.field("num_atoms", pa.int32()),
    pa.field("num_def_stereo", pa.int32()),
    pa.field("num_undef_stereo", pa.int32()),
    pa.field("num_def_double", pa.int32()),
    pa.field("num_undef_double", pa.int32()),
    pa.field("num_isotopic_atoms", pa.int32()),
    pa.field("fragments", pa.int32()),
    pa.field("num_tautomers", pa.int32()),
    pa.field("num_complexity", pa.int32()),
    pa.field("iupac_openeye", pa.string()),
    pa.field("iupac_cas", pa.string()),
    pa.field("iupac", pa.string()),
    pa.field("iupac_systematic", pa.string()),
    pa.field("iupac_traditional", pa.string()),
    pa.field("smiles", pa.string()),
    pa.field("set", pa.dictionary(pa.int32(), pa.string()))
]

schema = pa.schema(property_schema_fields)

There are 3 files for training, validation, and test data. Read and process the datasets in batch as the
 datasets are very large. We will be fine tuning instruction trained models like Qwen/Qwen2.5-0.5B-Instruct.  
 The training task should be of question and answer format where the question is of the format  
 f"What is the IUPAC name for the molecule {smiles}?" and the answer is of the format "It is {iupac}" 
 where {smiles} corresponds to the 'smiles' field in arrow schema and {iupac} comes from the 'iupac' 
 field in the arrow schema. The code to create the questions and answers should not be hardcoded for which 
 model is selected and use special tokens extracted from the model and tokenizer configuration.
use tokenizer.apply_chat_template() instead of hard coding the format of the question and answer. 
Please add logging, accuracy and perplexity metrics, command line arguments for the filename of the data and applicable TrainingArguments, 
and add an argument to limit the number of records loaded for training. Instead of using dataframes,
 just load the Datasets from the arrow files directly. Use a batched Dataset.map() to convert the 
 data into tokenized datasets for training and evaluation. It's important to minimize memory usage
   by batch processing in this script. 

"""
"""
Fine‑tune an instruction‑tuned causal‑LM on a SMILES → IUPAC Q&A task
and evaluate by *generation* (model sees only the question).

"""

# ───────────────────────────── main ────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--train_file", type=str, default='~/data/pubchem/arrow/cluster_100k_train.arrow', help="Path to the training Arrow file.")
parser.add_argument("--eval_file", type=str, default='~/data/pubchem/arrow/cluster_100k_eval.arrow', help="Path to the evaluation Arrow file.")
parser.add_argument("--test_file", type=str, default=None, help="Path to the test Arrow file (optional).")
parser.add_argument("--output_dir", type=str, default="~/results", help="Path to the output directory.")
# parser.add_argument("--model_name", default="microsoft/phi-4")
# parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
# parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
parser.add_argument("--model_name", default="Qwen/Qwen3-1.7B")
parser.add_argument("--max_records", default=3000, type=int)
parser.add_argument("--eval_limit",  type=int, default=10)
parser.add_argument(
    "--max_length",
    type=int,
    default=512,
    help="Maximum total sequence length for input tensors (prompt + label) during training or evaluation. "
         "Numerical relationship: len(prompt_tokens) + len(label_tokens) <= max_length. "
         "Example: If max_length=1024, prompt=800 tokens, label=300 tokens, label will be truncated so that (prompt + label) <= 1024."
)
parser.add_argument(
    "--max_label_len",
    type=int,
    default=512,
    help="Maximum length for the label (target/output) sequence in supervised training. "
         "Numerical relationship: len(label_tokens) <= max_label_len. "
         "Example: If max_label_len=256 and label=300 tokens, label will be truncated to 256 tokens. "
         "Note: The final combined length is still subject to max_length."
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=512,
    help="Maximum number of new tokens the model is allowed to generate during inference (generation). "
         "Numerical relationship: generated_tokens <= max_new_tokens. "
         "Example: If max_new_tokens=128, model will generate at most 128 new tokens for any prompt, regardless of input length. "
         "This parameter does not affect input tensor size, only the length of generated output during inference."
)

parser.add_argument("--num_beams", type=int, default=1)
parser.add_argument("--num_train_epochs", type=int, default=1)
parser.add_argument("--map_num_proc", type=int, default=os.cpu_count()-2,
                    help="Number of parallel processes to use in dataset.map")
parser.add_argument("--per_device_train_batch_size", type=int, default=2)
parser.add_argument("--per_device_eval_batch_size",  type=int, default=2)
parser.add_argument("--logging_steps", type=int, default=200)
parser.add_argument("--eval_steps",    type=int, default=400)
parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="number of steps during gradient accumulation. When using DeepSpeed, configure to use the same number of gradient accumulation step as in the DeepSpeed config")
parser.add_argument("--question_set", type=str, default="iupac_naming", choices=["iupac_naming", "molecular_properties", "all_properties"],
                    help="Type of question set to use for training")
parser.add_argument("--show_examples", action="store_true", help="Show example questions and answers and exit before training.")
parser.add_argument("--model_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"], 
                    help="Model precision, use float16 for Apple Silicon")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
# ensure left‑padding for causal generation and set pad to EOS
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                             torch_dtype=getattr(torch, args.model_dtype))


try:
    processor = PROCESSOR_CLASSES[args.question_set]()
except KeyError:
    raise ValueError(f"Unknown question set: {args.question_set}")

# ───────── load raw datasets ─────────
train_raw = load_arrow_dataset(args.train_file, args.max_records)
eval_raw  = load_arrow_dataset(args.eval_file, args.eval_limit or None)
test_raw  = load_arrow_dataset(args.test_file) if args.test_file else None

# Register a mode to display example Q&A without training
if args.show_examples:
    logging.info("Showing example questions and answers")
    processor.show_examples(eval_raw, tokenizer, args.eval_limit)
    sys.exit(0)

# ───────── tokenise datasets ─────────
with accelerator.main_process_first():
    # Prepare answers using QuestionSetProcessor
    train_answers = processor.prepare_answers(train_raw)
    eval_answers = processor.prepare_answers(eval_raw)
    test_answers = processor.prepare_answers(test_raw) if test_raw else None

    # Expand datasets to include all Q&A pairs
    logging.info("Expanding training dataset to include all Q&A pairs")
    expanded_train = processor.expand_dataset(train_raw, train_answers)
    logging.info(f"Expanded training dataset from {len(train_raw)} to {len(expanded_train)} examples")
    
    logging.info("Expanding evaluation dataset to include all Q&A pairs")
    expanded_eval = processor.expand_dataset(eval_raw, eval_answers)
    logging.info(f"Expanded evaluation dataset from {len(eval_raw)} to {len(expanded_eval)} examples")
    
    expanded_test = None
    if test_raw:
        logging.info("Expanding test dataset to include all Q&A pairs")
        expanded_test = processor.expand_dataset(test_raw, test_answers)
        logging.info(f"Expanded test dataset from {len(test_raw)} to {len(expanded_test)} examples")

    # Process each Q&A pair
    train_tok = expanded_train.map(
        lambda example: process_single_qa(
            tokenizer, example, args.max_length, is_train=True
        ),
        batched=False,
        num_proc=args.map_num_proc,
        remove_columns=["smiles", "question_id", "question_template", "answer", "assistant_template", "system_prompt"]
    )
    
    eval_tok = expanded_eval.map(
        lambda example: process_single_qa(
            tokenizer, example, args.max_length, max_label_len=args.max_label_len, is_train=False
        ),
        batched=False,
        num_proc=args.map_num_proc,
        remove_columns=["smiles", "question_id", "question_template", "answer", "assistant_template", "system_prompt"]
    )
    
    test_tok = None
    if expanded_test:
        test_tok = expanded_test.map(
            lambda example: process_single_qa(
                tokenizer, example, args.max_length, max_label_len=args.max_label_len, is_train=False
            ),
            batched=False,
            num_proc=args.map_num_proc,
            remove_columns=["smiles", "question_id", "question_template", "answer", "assistant_template", "system_prompt"]
        )

# ───────── FREE large raw datasets to save RAM ─────────
del train_raw, eval_raw, expanded_train
if test_raw:
    del test_raw

gc.collect()

# ───────── training setup ─────────
targs = Seq2SeqTrainingArguments(
    output_dir=str(Path(args.output_dir).expanduser()),
    eval_strategy="steps",
    eval_steps=args.eval_steps,
    batch_eval_metrics=True,  # necessary to avoid overflowing memory
    predict_with_generate=True,
    generation_max_length=args.max_new_tokens,
    generation_num_beams=args.num_beams,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    logging_steps=args.logging_steps,
    save_total_limit=2,
    metric_for_best_model="exact_match",
#    tf32=True,
    bf16=True,
    gradient_accumulation_steps=args.gradient_accumulation_steps
)

compute_metrics = compute_metrics_closure(tokenizer)

trainer = Trainer(
    model=model,
    args=targs,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

logging.info("Starting training …")
trainer.train()

logging.info("Generating validation predictions …")
val_metrics = trainer.evaluate(eval_dataset=eval_tok)
# labels = eval_tok["labels"]
# val_metrics = compute_metrics((val_preds, np.array(labels)))
logging.info("Validation metrics: %s", val_metrics)
val_preds = do_generation(args.num_beams, args.max_new_tokens, tokenizer, model, eval_tok)
if accelerator.is_main_process:
    show_examples(expanded_eval, val_preds, tokenizer, n=10)

if test_tok and test_tok is not None:
    test_metrics = trainer.evaluate(eval_dataset=test_tok)
    logging.info("Test metrics: %s", test_metrics)

    test_preds = do_generation(args.num_beams, args.max_new_tokens, tokenizer, model, test_tok)
    if accelerator.is_main_process:
        show_examples(expanded_test, test_preds, tokenizer, n=10)

trainer.save_model(str(Path(args.output_dir).expanduser()))
logging.info("Model saved to %s", args.output_dir)
