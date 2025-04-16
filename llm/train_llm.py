#!/usr/bin/env python
import argparse, logging, gc
from pathlib import Path

import numpy as np
import evaluate
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
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

System prompt: "You are a helpful chemistry professor."
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
            {"role": "user",
             "content": f"What is the IUPAC name for the molecule {s}?"},
            {"role": "assistant", "content": f"It is {i}"}
        ]
        for s, i in zip(smiles, iupac)
    ]
    enc = tok([tok.apply_chat_template(m) for m in msgs],
              truncation=True, padding="max_length", max_length=max_len)
    enc["labels"] = enc["input_ids"].copy()
    return enc


def build_eval_batch(tok, smiles, iupac,
                     max_prompt_len, max_label_len):
    user_prompts = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"What is the IUPAC name for the molecule {s}?"}
        ]
        for s in smiles
    ]
    prompt_enc = tok(
        [tok.apply_chat_template(m, add_generation_prompt=True)
         for m in user_prompts],
        truncation=True, padding="max_length", max_length=max_prompt_len,
    )

    ans_enc = tok([f"It is {i}" for i in iupac],
                  truncation=True, add_special_tokens=False,
                  max_length=max_label_len)

    pad = tok.pad_token_id
    labels = [
        ids + [pad] * (max_label_len - len(ids))
        for ids in ans_enc["input_ids"]
    ]
    prompt_enc["labels"] = (
        np.where(np.array(labels) == pad, -100, labels).tolist()
    )
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
    preds_txt = tokenizer.batch_decode(preds, skip_special_tokens=True)

    pad_id = tokenizer.pad_token_id
    labels_txt = tokenizer.batch_decode(
        np.where(labels == -100, pad_id, labels),
        skip_special_tokens=True
    )

    acc = exact_match.compute(
        predictions=[_norm(t) for t in preds_txt],
        references=[_norm(t) for t in labels_txt]
    )["exact_match"]
    return {"exact_match": acc}


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
parser.add_argument("--train_file", required=True)
parser.add_argument("--eval_file",  required=True)
parser.add_argument("--test_file")
parser.add_argument("--model_name", default="microsoft/phi-4")
parser.add_argument("--output_dir", default="~/results")
parser.add_argument("--max_records", type=int)
parser.add_argument("--eval_limit",  type=int, default=50)
parser.add_argument("--max_length",  type=int, default=512)
parser.add_argument("--max_label_len", type=int, default=64)
parser.add_argument("--max_new_tokens", type=int, default=32)
parser.add_argument("--num_beams", type=int, default=1)
parser.add_argument("--num_train_epochs", type=int, default=3)
parser.add_argument("--per_device_train_batch_size", type=int, default=8)
parser.add_argument("--per_device_eval_batch_size",  type=int, default=8)
parser.add_argument("--logging_steps", type=int, default=500)
parser.add_argument("--eval_steps",    type=int, default=1000)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model     = AutoModelForCausalLM.from_pretrained(args.model_name)

# ───────── load raw datasets ─────────
train_raw = load_arrow_dataset(args.train_file, args.max_records)
eval_raw  = load_arrow_dataset(args.eval_file, args.eval_limit or None)
test_raw  = load_arrow_dataset(args.test_file) if args.test_file else None

# ───────── tokenise datasets ─────────
train_tok = train_raw.map(
    lambda b: build_train_batch(tokenizer, b["smiles"], b["iupac"],
                                args.max_length),
    batched=True, batch_size=1000,
    remove_columns=["smiles", "iupac"]
)
eval_tok = eval_raw.map(
    lambda b: build_eval_batch(tokenizer, b["smiles"], b["iupac"],
                               args.max_length, args.max_label_len),
    batched=True, batch_size=1000,
    remove_columns=["smiles", "iupac"]
)
test_tok = None
if test_raw:
    test_tok = test_raw.map(
        lambda b: build_eval_batch(tokenizer, b["smiles"], b["iupac"],
                                   args.max_length, args.max_label_len),
        batched=True, batch_size=1000,
        remove_columns=["smiles", "iupac"]
    )

# ───────── FREE large raw datasets to save RAM ─────────
del train_raw, test_raw
gc.collect()

# ───────── training setup ─────────
targs = TrainingArguments(
    output_dir=str(Path(args.output_dir).expanduser()),
    evaluation_strategy="steps",
    eval_steps=args.eval_steps,
    predict_with_generate=True,
    generation_max_new_tokens=args.max_new_tokens,
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
trainer.train()

val_metrics = trainer.evaluate()
logging.info("Validation metrics: %s", val_metrics)
val_preds = trainer.predict(eval_tok).predictions
show_examples(eval_raw, val_preds, tokenizer, n=10)

if test_tok:
    test_metrics = trainer.evaluate(eval_dataset=test_tok)
    logging.info("Test metrics: %s", test_metrics)
    test_preds = trainer.predict(test_tok).predictions
    show_examples(eval_raw if test_raw is None else test_raw,
                  test_preds, tokenizer, n=10)

trainer.save_model(args.output_dir)
logging.info("Model saved to %s", args.output_dir)