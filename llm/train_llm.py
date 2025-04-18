#!/usr/bin/env python
from __future__ import annotations
import argparse, logging, gc
from pathlib import Path
import os

import numpy as np
import torch
import evaluate
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    Seq2SeqTrainingArguments,
)


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

SYSTEM_PROMPT = "You are a helpful chemistry professor."

# ─────────────────────────── data helper ──────────────────────────────
def load_arrow_dataset(path: str, limit: int | None = None) -> Dataset:
    ds = Dataset.from_file(str(Path(path).expanduser()))
    if limit and len(ds) > limit:
        ds = ds.select(range(limit))
    return ds


# ───────────────────────── tokenisation helpers ───────────────────────
def build_train_batch(tok, smiles, iupac, max_len):
    msgs = [
        [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": f"What is the IUPAC name for the molecule {s}?"},
            {"role": "assistant", "content": f"{i}"}
        ]
        for s, i in zip(smiles, iupac)
    ]
    # Step 1: Get prompt strings
    prompts = [tok.apply_chat_template(m, tokenize=False) for m in msgs]

    # Step 2: Tokenize prompts
    enc = tok(prompts, padding="max_length", truncation=True,
              max_length=max_len, return_tensors="np")
    enc["labels"] = enc["input_ids"].copy()
    return enc

def build_eval_batch(tok, smiles, iupac,
                     max_prompt_len, max_label_len):
    user_msgs = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"What is the IUPAC name for the molecule {s}?"}
        ]
        for s in smiles
    ]

    # Step 1: Prompt strings
    prompts = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
               for m in user_msgs]

    # Step 2: Tokenize prompts
    prompt_enc = tok(prompts, padding="max_length", truncation=True,
                     max_length=max_prompt_len, return_tensors="np")

    # Labels: tokenize only the answer
    ans_enc = tok([f"{i}" for i in iupac],
                  truncation=True, add_special_tokens=False,
                  max_length=max_label_len, return_tensors="np")

    # Build labels matching the full sequence length, ignoring prompt tokens
    answers = ans_enc["input_ids"].tolist()
    labels_full = []
    for answer in answers:
        # Initialize all positions to ignore_index (-100)
        label = [-100] * max_prompt_len
        # Right-align the answer tokens at the end of the sequence
        offset = max_prompt_len - len(answer)
        label[offset:] = answer
        labels_full.append(label)
    prompt_enc["labels"] = labels_full
    return prompt_enc


# ─────────────────────────── metrics & helpers ────────────────────────
exact_match = evaluate.load("exact_match")


def _norm(s: str) -> str:
    s = s.strip().lower()
    if s.startswith("it is"):
        s = s[5:].strip()
    return s.rstrip(".")


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # Convert logits to predicted token IDs if needed
    if preds.ndim == 3:
        preds_ids = np.argmax(preds, axis=-1)
    else:
        preds_ids = preds

    # Extract only answer tokens based on label mask
    preds_answer_ids = []
    labels_answer_ids = []
    for pred_row, label_row in zip(preds_ids, labels):
        mask = label_row != -100
        preds_answer_ids.append(pred_row.tolist())
        labels_answer_ids.append(label_row[mask].tolist())

    # Decode the answer sequences
    preds_txt = [tokenizer.decode(ids, skip_special_tokens=True).strip()
                 for ids in preds_answer_ids]
    labels_txt = [tokenizer.decode(ids, skip_special_tokens=True).strip()
                  for ids in labels_answer_ids]

    # Compute exact match accuracy on the IUPAC names
    matches = []
    for pred, label in zip(preds_txt, labels_txt):
        if labels_txt in preds_txt:
            matches.append(1.0)
        else:
            matches.append(0.0)
    return {"exact_match": sum(matches)/len(matches)}


def show_examples(raw_ds, preds, tok, n=10):
    print("\n──────── First {} examples ────────".format(n))
    for i in range(min(n, len(raw_ds))):
        q = f"What is the IUPAC name for the molecule {raw_ds[i]['smiles']}?"
        gt = f"It is {raw_ds[i]['iupac']}"
        pd = tok.decode(preds[i], skip_special_tokens=True).strip()
        print(f"\n#{i}")
        print("Q :", q)
        print("GT:", gt)
        print("PD:", pd)


# ───────────────────────────── main ────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--train_file", type=str, default='~/data/pubchem/arrow/test_train.arrow', help="Path to the training Arrow file.")
parser.add_argument("--eval_file", type=str, default='~/data/pubchem/arrow/test_valid.arrow', help="Path to the evaluation Arrow file.")
parser.add_argument("--test_file", type=str, default=None, help="Path to the test Arrow file (optional).")
parser.add_argument("--output_dir", type=str, default="~/results", help="Path to the output directory.")
parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
parser.add_argument("--max_records", default=100, type=int)
parser.add_argument("--eval_limit",  type=int, default=10)
parser.add_argument("--max_length",  type=int, default=128)
parser.add_argument("--max_label_len", type=int, default=128)
parser.add_argument("--max_new_tokens", type=int, default=128)
parser.add_argument("--num_beams", type=int, default=1)
parser.add_argument("--num_train_epochs", type=int, default=1)
parser.add_argument("--map_num_proc", type=int, default=os.cpu_count()-2,
                    help="Number of parallel processes to use in dataset.map")
parser.add_argument("--per_device_train_batch_size", type=int, default=24)
parser.add_argument("--per_device_eval_batch_size",  type=int, default=10)
parser.add_argument("--logging_steps", type=int, default=500)
parser.add_argument("--eval_steps",    type=int, default=1)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
# ensure left‑padding for causal generation and set pad to EOS
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(args.model_name)

# ───────── load raw datasets ─────────
train_raw = load_arrow_dataset(args.train_file, args.max_records)
eval_raw  = load_arrow_dataset(args.eval_file, args.eval_limit or None)
test_raw  = load_arrow_dataset(args.test_file) if args.test_file else None

# ───────── tokenise datasets ─────────
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
del train_raw, test_raw
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
)

trainer = Trainer(
    model=model,
    args=targs,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

logging.info("Starting training …")
# trainer.train()

logging.info("Generating validation predictions …")
eval_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])
val_input_ids = eval_tok["input_ids"].to(model.device)
val_attention_mask = eval_tok["attention_mask"].to(model.device)
# drop the final token (EOS or pad) so generation will produce new tokens
val_input_ids      = val_input_ids[:, :-1]
val_attention_mask = val_attention_mask[:, :-1]

val_preds = model.generate(
    input_ids=val_input_ids,
    attention_mask=val_attention_mask,
    max_new_tokens=args.max_new_tokens,
    num_beams=args.num_beams,
)
show_examples(eval_raw, val_preds, tokenizer, n=10)

# manually compute metrics
labels = eval_tok["labels"]
val_metrics = compute_metrics((val_preds, np.array(labels)))
logging.info("Validation metrics: %s", val_metrics)

if test_tok:
    test_metrics = trainer.evaluate(eval_dataset=test_tok)
    logging.info("Test metrics: %s", test_metrics)
    test_inputs = test_tok["input_ids"]
    test_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])
    test_input_ids = test_tok["input_ids"].to(model.device)
    test_attention_mask = test_tok["attention_mask"].to(model.device)
    # drop the final token so generation will actually happen
    test_input_ids      = test_input_ids[:, :-1]
    test_attention_mask = test_attention_mask[:, :-1]
    
    test_preds = model.generate(
        input_ids=test_input_ids,
        attention_mask=test_attention_mask,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )
    show_examples(eval_raw if test_raw is None else test_raw,
                  test_preds, tokenizer, n=10)

trainer.save_model(args.output_dir)
logging.info("Model saved to %s", args.output_dir)
