import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.compute as pc
import sys
import argparse
from pathlib import Path

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate questions JSONL from raw data + YAML config")
    p.add_argument("--input", default="~/data/pubchem/arrow/pubchem_sorted.arrow", help="Path to raw tabular file (csv, tsv, jsonl, parquet)")
    p.add_argument("--output", default="~/data/pubchem/arrow/pubchem_best.arrow", help="Destination JSONL file (one Q-A per line)")
    p.add_argument("--filter", default="~/data/pubchem/arrow/pccompound_best.txt", help="file containing list of CIDs to filter, one per line")
    p.add_argument("--valid-frac", type=float, default=0.05, help="Validation split fraction (before expansion)")
    p.add_argument("--test-frac", type=float, default=0.05, help="Test split fraction (before expansion)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def _assign_splits(table: pa.Table, valid_frac: float, test_frac: float, seed: int) -> pa.Table:
    import numpy as np
    assert valid_frac + test_frac < 1.0, "Split fractions too large"
    rng = np.random.default_rng(seed)
    choices = rng.choice(
        ["test", "valid", "train"],
        size=table.num_rows,
        p=[test_frac, valid_frac, 1.0 - valid_frac - test_frac],
    )
    split_arr = pa.array(choices, type=pa.string())
    return table.append_column("split", split_arr)

def main() -> None:
    args = _parse_args()
    ARROW_PATH = Path(args.input).expanduser()
    OUTPUT_PATH = Path(args.output).expanduser()
    CID_TXT_PATH = Path(args.filter).expanduser()

    # 1. Load CIDs from text file
    cid_set = None
    if CID_TXT_PATH != '':
        with open(CID_TXT_PATH, "r") as f:
            cid_set = set(line.strip() for line in f if line.strip())

    # 2. Load Arrow table (random access, memory-mapped)
    reader = ipc.RecordBatchFileReader(pa.memory_map(ARROW_PATH, "rb"))
    table = reader.read_all()

    # 3. Subset table by CIDs
    if "cid" not in table.column_names:
        sys.exit("Error: 'cid' column not found in Arrow table.")
    cid_col = table.column("cid")
    # Convert Arrow column to Python strings for set lookup
    if cid_set is not None:
        if pa.types.is_integer(cid_col.type):
            mask = [str(cid_col[i].as_py()) in cid_set for i in range(len(cid_col))]
        else:
            mask = [cid_col[i].as_py() in cid_set for i in range(len(cid_col))]
        table = table.filter(pa.array(mask))

    # ignore records with empty IUPAC names
    if table.column("iupac"):
        table = table.filter(pc.equal(table['iupac'], ''))

    table = _assign_splits(table, args.valid_frac, args.test_frac, args.seed)

    # 4. Write subset to Arrow file (random access format)
    with pa.OSFile(OUTPUT_PATH, "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)

    print(f"Wrote subset to {OUTPUT_PATH} with {table.num_rows} rows.")
