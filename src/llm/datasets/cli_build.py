#!/usr/bin/env python
"""CLI: build tokenised HF datasets from *questions.jsonl*.

Example:

    python -m llm.datasets.cli_build \
        --questions questions.jsonl \
        --tokenizer openai/gpt2 \
        --output data/chem_prop_ds
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from transformers import AutoTokenizer
from datasets import DatasetDict

from .preprocess import (
    load_questions_jsonl,
    split_dataset,
    split_by_column,
    tokenise_dataset_dict,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build HF tokenised dataset from questions.jsonl")
    p.add_argument("--questions", required=True, help="Path to questions.jsonl (from cli_generate)")
    p.add_argument("--tokenizer", required=True, help="HF tokenizer name/path")
    p.add_argument("--output", required=True, help="Output directory to write dataset files (saved as HF Arrow)")
    p.add_argument("--max-length", type=int, default=4096, help="Max prompt length")
    p.add_argument("--max-label-len", type=int, default=1024)
    p.add_argument("--valid-frac", type=float, default=0.025)
    p.add_argument("--test-frac", type=float, default=0.025)
    p.add_argument("--num-proc", type=int, default=None, help="Parallelism for .map()")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    q_path = Path(args.questions)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_questions_jsonl(q_path)
    if "split" in ds.column_names:
        split_ds = split_by_column(ds)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # Attempt to read system_prompt from YAML with same stem as input questions
    system_prompt: str | None = None
    try:
        stem = q_path.stem  # e.g. molecular_properties_questions → molecular_properties
        yaml_path = q_path.with_name(stem + ".yaml")
        if yaml_path.exists():
            import yaml as _yaml
            with open(yaml_path, "r", encoding="utf-8") as _f:
                _cfg = _yaml.safe_load(_f)
            system_prompt = _cfg.get("system_prompt")
    except Exception as _exc:
        logger.debug("Could not load system_prompt from YAML: %s", _exc)

    full_tok, minimal_tok = tokenise_dataset_dict(
        split_ds,
        tokenizer,
        max_length=args.max_length,
        max_label_len=args.max_label_len,
        num_proc=args.num_proc,
        system_prompt=system_prompt,
    )

    logger.info("Saving tokenised datasets to %s", out_dir)
    full_tok.save_to_disk(str(out_dir / "full"))
    minimal_tok.save_to_disk(str(out_dir / "minimal"))


if __name__ == "__main__":
    main()
