#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path

import numpy as np
import evaluate
from datasets import Dataset


SYSTEM_PROMPT = "You are a helpful chemistry professor. Be concise. Only provide one chemical name as the answer and place it between <|box_start|> and <|box_end|>."

# ─────────────────────────── data helper ──────────────────────────────
def load_arrow_dataset(path: str, limit: int | None = None) -> Dataset:
    ds = Dataset.from_file(str(Path(path).expanduser()))
    if limit and limit > 0 and len(ds) > limit:
        ds = ds.select(range(limit))
    return ds


# ───────────────────────── tokenisation helpers ───────────────────────
def build_train_batch(tok, smiles, iupac, max_len):
    msgs = [
        [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": f"What is the IUPAC name for the molecule {s}?"},
            {"role": "assistant", "content": f"<|box_start|>{i}<|box_end|>"}
        ]
        for s, i in zip(smiles, iupac)
    ]
    # Step 1: Get prompt strings
    prompts = [tok.apply_chat_template(m, tokenize=False) for m in msgs]

    # Step 2: Tokenize prompts
    enc = tok(prompts, padding="max_length", truncation=True,
              max_length=max_len, return_tensors="np")

    # Prepare answer token IDs without special tokens
    answers_ids = [tok(text, add_special_tokens=False)["input_ids"] for text in iupac]
    input_ids_list = enc["input_ids"].tolist()

    labels_full = []
    for row_ids, ans_ids, smile in zip(input_ids_list, answers_ids, smiles):
        # Find the start index of the answer tokens within the input_ids sequence
        start_idx = -1
        for i in range(len(row_ids) - len(ans_ids) + 1):
            if row_ids[i : i + len(ans_ids)] == ans_ids[:len(row_ids)-i]:
                start_idx = i
                break
        label = [-100] * len(row_ids)
        if start_idx >= 0:
            # Build label row: mask (-100) everywhere except the answer span
            for j, tok_id in enumerate(ans_ids):
                label[start_idx + j] = tok_id
        else:
            print(f"Warning: Answer not found in input_ids for SMILES: {smile}")
        labels_full.append(label)

    enc["labels"] = labels_full
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
    ans_enc = tok([f"<|box_start|>{i}<|box_end|>" for i in iupac],
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

def compute_metrics_closure(tokenizer):
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
            if label in preds_txt:
                matches.append(1.0)
            else:
                matches.append(0.0)
        return {"exact_match": sum(matches)/len(matches)}
    return compute_metrics


def show_examples(raw_ds, preds, tok, n=10):
    print("\n──────── First {} examples ────────".format(n))
    for i in range(0, len(raw_ds), len(raw_ds)//n):
        q = f"What is the IUPAC name for the molecule {raw_ds[i]['smiles']}?"
        gt = f"It is {raw_ds[i]['iupac']}"
        pd = tok.decode(preds[i], skip_special_tokens=True).strip()
        print(f"\n#{i}")
        print("Q :", q)
        print("GT:", gt)
        print("PD:", pd)

def do_generation(num_beams, max_new_tokens, tokenizer, model, data):
    data.set_format(type="torch", columns=["input_ids", "attention_mask"])
    input_ids = data["input_ids"].to(model.device)
    attention_mask = data["attention_mask"].to(model.device)
# drop the final token (EOS or pad) so generation will produce new tokens
    input_ids      = input_ids[:, :-1]
    attention_mask = attention_mask[:, :-1]

    preds = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=True,
        top_p=0.8,
        temperature=0.7,
        top_k=20,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
        length_penalty=1.5,
    #    stopping_criteria=StoppingCriteriaList([EosTokenCriteria(tokenizer.eos_token_id)])
    )
    
    return preds