import pyarrow as pa
import pyarrow.ipc as ipc
import sys

# Input files (adjust paths as needed)
ARROW_PATH = "pubchem_sorted.arrow"
CID_TXT_PATH = "pccompound_result.txt"
OUTPUT_PATH = "pubchem_subset.arrow"

# 1. Load CIDs from text file
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
if pa.types.is_integer(cid_col.type):
    mask = [str(cid_col[i].as_py()) in cid_set for i in range(len(cid_col))]
else:
    mask = [cid_col[i].as_py() in cid_set for i in range(len(cid_col))]
subset_table = table.filter(pa.array(mask))

# 4. Write subset to Arrow file (random access format)
with pa.OSFile(OUTPUT_PATH, "wb") as sink:
    with ipc.new_file(sink, subset_table.schema) as writer:
        writer.write_table(subset_table)

print(f"Wrote subset to {OUTPUT_PATH} with {subset_table.num_rows} rows.")
