import argparse
import logging

import pyarrow as pa
import polars as pl
import numpy as np
from datasets.arrow_writer import ArrowWriter 

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_prepare_data(arrow_path, output_prefix, max_records=None, p=None):
    if p is None:
        p = (0.95, 0.025, 0.025)
    writers = {"train": ArrowWriter(path=f"{output_prefix}_train.arrow"),
                "valid": ArrowWriter(path=f"{output_prefix}_valid.arrow"),
                "test": ArrowWriter(path=f"{output_prefix}_test.arrow")}
    total_loaded = 0

    with pa.ipc.RecordBatchFileReader(pa.OSFile(str(arrow_path), 'r')) as reader:
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            df = pl.from_arrow(batch)
            for row in df.iter_rows(named=True):
                if max_records and total_loaded >= max_records:
                    break
                split = str(np.random.choice(('train','valid','test'), p=p))
                writers[split].write({"smiles": row['smiles'],'iupac': row['iupac']})
                total_loaded += 1
            if max_records and total_loaded >= max_records:
                    break
    for writer in writers.values():
        writer.finalize()


def main():
    parser = argparse.ArgumentParser(description="make huggingface datasets")
    parser.add_argument("--arrow_file", type=str, default='/home/lyg/source/l1nus/etl/pubchem/pubchem/pubchem.arrow', help="Path to the Arrow file.")
    parser.add_argument("--output_prefix", type=str, default="test")
    parser.add_argument("--max_records", type=int, default=None, help="Limit the number of records loaded.")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    logger.info("Loading and preparing dataset...")
    load_and_prepare_data(args.arrow_file, args.output_prefix, max_records=args.max_records, p=(0.95,0.025,0.025))


if __name__ == "__main__":
    main()


"""
 from datasets.arrow_writer import ArrowWriter                                                                                                                                                   

In [2]: with ArrowWriter(path="tmp.arrow") as writer: 
   ...:     writer.write({"a": 1}) 
   ...:     writer.write({"a": 2}) 
   ...:     writer.write({"a": 3}) 
   ...:     writer.finalize() 
   ...:                                                                                                                                                                                                 

In [3]: from datasets import Dataset                                                                                                                                                                    

In [4]: ds = Dataset.from_file("tmp.arrow") 

or 
batch = {"a": [4, 5, 6]}
writer.write_batch(batch

"""