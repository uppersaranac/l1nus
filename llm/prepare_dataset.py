import argparse
import logging

import pyarrow as pa
import polars as pl
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    set_seed,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def load_and_prepare_data(arrow_path, max_records=None):
    data = {"train": [], "valid": [], "test": []}
    total_loaded = 0

    df = pl.read_ipc(arrow_path, memory_map=True, columns=['smiles','iupac'])

    for row in df.iter_rows(named=True):
        if max_records and total_loaded >= max_records:
            break
        split = str(np.random.choice(('train','valid','test'), p=(0.95,0.025,0.025)))
        if isinstance(split, dict):
            split = list(split.values())[0]  # handle dictionary field
        question = f"What is the IUPAC name for the molecule {row['smiles']}?"
        answer = f"It is {row['iupac']}"
        full_text = f"{question} {answer}"
        data[split].append({"text": full_text})
        total_loaded += 1

    dataset = DatasetDict({
        split: Dataset.from_list(samples)
        for split, samples in data.items() if samples
    })
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Train Phi-4 on IUPAC chemical naming task.")
    parser.add_argument("--arrow_file", type=str, default='/home/lyg/source/l1nus/etl/pubchem/pubchem/pubchem.arrow', help="Path to the Arrow file.")
    parser.add_argument("--output_file", type=str, default="test")
    parser.add_argument("--max_records", type=int, default=None, help="Limit the number of records loaded.")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    logger.info("Loading and preparing dataset...")
    dataset = load_and_prepare_data(args.arrow_file, args.max_records)

    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-4", trust_remote_code=True)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=args.max_length, padding="max_length"
        )

    logger.info("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_datasets.shuffle(seed=42)
    tokenized_datasets.save_to_disk(args.output_file)



if __name__ == "__main__":
    main()