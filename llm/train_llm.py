#!/usr/bin/env python
from __future__ import annotations
import argparse, logging, gc
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

from llm_apis import *

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
parser.add_argument("--max_length",  type=int, default=1024)
parser.add_argument("--max_label_len", type=int, default=1024)
parser.add_argument("--max_new_tokens", type=int, default=1024)
parser.add_argument("--num_beams", type=int, default=1)
parser.add_argument("--num_train_epochs", type=int, default=1)
parser.add_argument("--map_num_proc", type=int, default=os.cpu_count()-2,
                    help="Number of parallel processes to use in dataset.map")
parser.add_argument("--per_device_train_batch_size", type=int, default=4)
parser.add_argument("--per_device_eval_batch_size",  type=int, default=4)
parser.add_argument("--logging_steps", type=int, default=200)
parser.add_argument("--eval_steps",    type=int, default=200)
parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="number of steps during gradient accumulation. When using DeepSpeed, configure to use the same number of gradient accumulation step as in the DeepSpeed config")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
# ensure left‑padding for causal generation and set pad to EOS
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                             torch_dtype=torch.bfloat16)

# ───────── load raw datasets ─────────
train_raw = load_arrow_dataset(args.train_file, args.max_records)
eval_raw  = load_arrow_dataset(args.eval_file, args.eval_limit or None)
test_raw  = load_arrow_dataset(args.test_file) if args.test_file else None

# ───────── tokenise datasets ─────────
with accelerator.main_process_first():
    train_tok = train_raw.map(
        lambda b: build_train_batch(tokenizer, b["smiles"], b["iupac"],
                                    args.max_length),
        batched=True, batch_size=1000,
        remove_columns=["smiles", "iupac"],
        num_proc=args.map_num_proc
    )
    eval_tok = eval_raw.map(
        lambda b: build_eval_batch(tokenizer, b["smiles"], b["iupac"],
                                args.max_length, args.max_label_len),
        batched=True, batch_size=1000,
        remove_columns=["smiles", "iupac"],
        num_proc=args.map_num_proc
    )
    test_tok = None
    if test_raw:
        test_tok = test_raw.map(
            lambda b: build_eval_batch(tokenizer, b["smiles"], b["iupac"],
                                    args.max_length, args.max_label_len),
            batched=True, batch_size=1000,
            remove_columns=["smiles", "iupac"],
            num_proc=args.map_num_proc
        )

# ───────── FREE large raw datasets to save RAM ─────────
del train_raw
gc.collect()

# ───────── training setup ─────────
targs = Seq2SeqTrainingArguments(
    output_dir=str(Path(args.output_dir).expanduser()),
    eval_strategy="steps",
    eval_steps=args.eval_steps,
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
    show_examples(eval_raw, val_preds, tokenizer, n=10)

if test_tok and test_tok is not None:
    test_metrics = trainer.evaluate(eval_dataset=test_tok)
    logging.info("Test metrics: %s", test_metrics)

    test_preds = do_generation(args.num_beams, args.max_new_tokens, tokenizer, model, test_tok)
    if accelerator.is_main_process:
        show_examples(test_raw, test_preds, tokenizer, n=10)

trainer.save_model(str(Path(args.output_dir).expanduser()))
logging.info("Model saved to %s", args.output_dir)
