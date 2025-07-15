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
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.compute as pc
import pyarrow.json as pajson
import pyarrow.parquet as pq

from llm.questions.generators import GenerationConfig, QuestionGenerator
from llm.questions.processors import PROCESSOR_CLASSES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_table(path: Path, limit: int | None = None) -> pa.Table:
    """
    Load tabular data as a **memory-mapped** Arrow Table.

    Supported extensions:
      • .arrow   Arrow IPC random-access file (zero-copy mmap)
      • .parquet Parquet file
      • .csv     Comma-separated values
      • .tsv     Tab-separated values
      • .jsonl/.json Newline-delimited JSON

    :param path: Path to the input file.
    :type path: Path
    :param limit: Optionally limit the number of records.
    :type limit: int or None
    :return: Arrow Table containing the data.
    :rtype: pa.Table
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
    """
    Parse command-line arguments for question generation CLI.

    :return: Parsed arguments namespace.
    :rtype: argparse.Namespace
    """
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

    p.add_argument(
        "--filter-column",
        type=str,
        default="num_atoms",
        help="Column name to filter by numeric range (default: num_atoms)",
    )
    p.add_argument(
        "--filter-min",
        type=float,
        default=None,
        help="Minimum value (inclusive) for numeric filtering (default: no lower bound)",
    )
    p.add_argument(
        "--filter-max",
        type=float,
        default=None,
        help="Maximum value (inclusive) for numeric filtering (default: no upper bound)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Main entry point for the question generation CLI.
    """
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

    # Numeric range filtering using pyarrow.compute
    filter_col = args.filter_column
    filter_min = args.filter_min
    filter_max = args.filter_max
    if filter_min is not None or filter_max is not None:
        if filter_col not in table.column_names:
            logger.error(f"Column '{filter_col}' not found in table for filtering.")
            sys.exit(1)
        col = table[filter_col]
        mask = None
        if filter_min is not None and filter_max is not None:
            mask = pc.and_kleene( # type: ignore
                pc.greater_equal(col, pa.scalar(filter_min)), # type: ignore
                pc.less_equal(col, pa.scalar(filter_max)) # type: ignore
            )
        elif filter_min is not None:
            mask = pc.greater_equal(col, pa.scalar(filter_min)) # type: ignore
        elif filter_max is not None:
            mask = pc.less_equal(col, pa.scalar(filter_max)) # type: ignore
        table = table.filter(mask)
        logger.info(f"Filtered table on column '{filter_col}' with min={filter_min}, max={filter_max}. Remaining rows: {table.num_rows}")
    logger.info("Loaded %d records", table.num_rows)

    logger.info("Parsing YAML config %s", args.config)
    cfg = GenerationConfig.from_yaml(args.config)

    # Determine which QuestionSetProcessor (if any) to apply
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
        result = proc.prepare_answers(table)
        answers, mask = result
        mask_array = pa.array(mask)
        table = table.filter(mask_array)
        # Also filter the answers to match
        for col, values in answers.items():
            filtered_values = [v for v, m in zip(values, mask) if m]
            if col in table.column_names:
                table = table.remove_column(table.column_names.index(col))
            table = table.append_column(col, pa.array(filtered_values))

        logger.info("Computed answer columns via %s", qs_name)

    gen = QuestionGenerator(cfg)
    logger.info("Generating questions…")
    n_written = gen.generate_jsonl(table, output_path)
    logger.info("Wrote %d Q-A records to %s", n_written, output_path)


if __name__ == "__main__":
    main()
