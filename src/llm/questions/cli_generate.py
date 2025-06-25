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

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import pyarrow.json as pajson

from llm.questions.generators import GenerationConfig, QuestionGenerator
from llm.questions.processors import PROCESSOR_CLASSES

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_table(path: Path, limit: int | None = None) -> pa.Table:
    """Load tabular data as a **memory-mapped** Arrow Table.

    Supported extensions:
      • .arrow   – Arrow IPC random-access file (zero-copy mmap)
      • .parquet – Parquet file
      • .csv     – Comma-separated values
      • .tsv     – Tab-separated values
      • .jsonl/.json – Newline-delimited JSON
    """
    suffix = path.suffix.lower()

    if suffix == ".arrow":
        # Zero-copy read via memory mapping
        mmap = pa.memory_map(str(path), "r")
        # TODO: read random-access Arrow file
        reader = pa.ipc.RecordBatchFileReader(mmap)
        table = reader.read_all()
    elif suffix == ".parquet":
        table = pq.read_table(str(path))
    elif suffix in {".csv", ".tsv"}:
        parse_opts = pacsv.ParseOptions(delimiter="," if suffix == ".csv" else "\t")
        table = pacsv.read_csv(str(path), parse_options=parse_opts)
    elif suffix in {".jsonl", ".json"}:
        table = pajson.read_json(str(path), read_options=pajson.ReadOptions(newlines_in_values=False))
    else:
        raise ValueError(f"Unsupported input format: {suffix}")

    if limit is not None:
        table = table.slice(0, limit)
    return table


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate questions JSONL from raw data + YAML config")
    p.add_argument("--input", required=True, help="Path to raw tabular file (csv, tsv, jsonl, parquet, arrow)")
    p.add_argument("--config", required=True, help="YAML file describing question templates and system_prompt")
    p.add_argument("--output", required=True, help="Destination JSONL file (one Q-A per line)")
    p.add_argument("--limit", type=int, default=None, help="Optionally limit number of raw records for quick runs")
    p.add_argument(
        "--keep-null-columns",
        action="store_true",
        default=False,
        help="Keep columns containing null values (by default, columns with any nulls are dropped)",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()

    logger.info("Loading raw data from %s", input_path)
    table = _load_table(input_path, limit=args.limit)
    # Drop columns with any nulls unless user requests to keep them
    # the Huggingface json load_data does not like nulls
    if not args.keep_null_columns:
        before_cols = len(table.column_names)
        # Find columns with any nulls
        null_cols = [col for col in table.column_names if table.column(col).null_count > 0]
        if null_cols:
            logger.info(f"Dropping columns with nulls: {null_cols}")
            for col in null_cols:
                table = table.remove_column(table.column_names.index(col))
            logger.info(f"Dropped {len(null_cols)} columns with nulls (from {before_cols} to {len(table.column_names)})")
    logger.info("Loaded %d records", table.num_rows)

    logger.info("Parsing YAML config %s", args.config)
    cfg = GenerationConfig.from_yaml(args.config)

    # Determine which QuestionSetProcessor (if any) to apply
    from llm.questions.processors import PROCESSOR_CLASSES
    qs_name: str | None = cfg.question_set
    if qs_name is None:
        logger.error("No question set specified or inferred. Please set 'question_set' in the YAML config or use a known config filename.")
        sys.exit(1)

    if qs_name:
        try:
            proc_cls = PROCESSOR_CLASSES[qs_name]
        except KeyError:
            raise SystemExit(f"Unknown question set: {qs_name}")

        proc = proc_cls()
        # Create dataset-like mapping (lists) from Arrow columns
        ds_like = {col: table.column(col).to_pylist() for col in table.column_names}
        answers = proc.prepare_answers(ds_like)
        # Append each answer list as a new Arrow column (overwrite if exists)
        for col, values in answers.items():
            if col in table.column_names:
                table = table.remove_column(table.column_names.index(col))
            table = table.append_column(col, pa.array(values))
        logger.info("Computed answer columns via %s", qs_name)


    gen = QuestionGenerator(cfg)
    logger.info("Generating questions…")
    n_written = gen.generate_jsonl(table, output_path)
    logger.info("Wrote %d Q-A records to %s", n_written, output_path)


if __name__ == "__main__":
    main()
