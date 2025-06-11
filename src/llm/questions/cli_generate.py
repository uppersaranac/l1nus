#!/usr/bin/env python
"""CLI entry-point: generic question generation.

Example:

    python -m llm.questions.cli_generate \
        --input data/raw.csv \
        --config configs/qgen.yaml \
        --output questions.jsonl
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

from .generators import GenerationConfig, QuestionGenerator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_dataframe(path: Path, limit: int | None = None) -> pd.DataFrame:
    """Load tabular data using pandas based on file extension."""
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        df = pd.read_csv(path, sep="," if suffix == ".csv" else "\t")
    elif suffix in {".jsonl", ".json"}:
        df = pd.read_json(path, lines=True)
    elif suffix in {".parquet"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported input format: {suffix}")

    if limit is not None:
        df = df.head(limit)
    return df


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate questions JSONL from raw data + YAML config")
    p.add_argument("--input", required=True, help="Path to raw tabular file (csv, tsv, jsonl, parquet)")
    p.add_argument("--config", required=True, help="YAML file describing question templates and system_prompt")
    p.add_argument("--output", required=True, help="Destination JSONL file (one Q-A per line)")
    p.add_argument("--limit", type=int, default=None, help="Optionally limit number of raw records for quick runs")
    p.add_argument("--question-set", type=str, default=None, help="Name of predefined QuestionSetProcessor to compute answers (e.g., 'molecular_properties')")
    p.add_argument("--valid-frac", type=float, default=0.025, help="Validation split fraction (before expansion)")
    p.add_argument("--test-frac", type=float, default=0.025, help="Test split fraction (before expansion)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _assign_splits(df, valid_frac: float, test_frac: float, seed: int):
    import numpy as np
    assert valid_frac + test_frac < 1.0, "Split fractions too large"
    rng = np.random.default_rng(seed)
    choices = rng.choice(
        ["test", "valid", "train"],
        size=len(df),
        p=[test_frac, valid_frac, 1.0 - valid_frac - test_frac],
    )
    df["split"] = choices
    return df


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    logger.info("Loading raw data from %s", input_path)
    df = _load_dataframe(input_path, limit=args.limit)
    logger.info("Loaded %d records", len(df))

    logger.info("Parsing YAML config %s", args.config)
    cfg = GenerationConfig.from_yaml(args.config)

    # If user specifies a built-in question set, compute answers using its processor
    if args.question_set:
        try:
            from llm.questions.processors import PROCESSOR_CLASSES
            proc_cls = PROCESSOR_CLASSES[args.question_set]
        except KeyError:
            raise SystemExit(f"Unknown question set: {args.question_set}")

        proc = proc_cls()
        # Prepare answers expects a dataset-like dict with list values
        ds_like = {col: df[col].tolist() for col in df.columns}
        answers = proc.prepare_answers(ds_like)
        # Add each answer list as a new column in the dataframe
        for col, values in answers.items():
            df[col] = values
        logger.info("Computed answer columns via %s", args.question_set)

    df = _assign_splits(df, args.valid_frac, args.test_frac, args.seed)
    logger.info(
        "Split counts – train: %d, valid: %d, test: %d",
        (df["split"] == "train").sum(),
        (df["split"] == "valid").sum(),
        (df["split"] == "test").sum(),
    )

    gen = QuestionGenerator(cfg)
    logger.info("Generating questions…")
    n_written = gen.generate_jsonl(df, output_path)
    logger.info("Wrote %d Q-A records to %s", n_written, output_path)


if __name__ == "__main__":
    main()
