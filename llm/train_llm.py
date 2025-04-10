import argparse
import logging
import math
import os

import pyarrow.dataset as ds
import pyarrow as pa
import polars as pl
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
import evaluate

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Evaluation metrics
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    mask = labels != -100
    correct = (predictions == labels) & mask
    acc = correct.sum() / mask.sum()
    return {
        "accuracy": acc.item(),
        "perplexity": math.exp(eval_pred.loss) if hasattr(eval_pred, 'loss') else float('inf')
    }

def load_and_prepare_data(arrow_path, max_records=None):
    with pa.ipc.RecordBatchStreamReader(arrow_path) as arrow_dataset:
        data = {"train": [], "valid": [], "test": []}
        total_loaded = 0

        for batch in arrow_dataset.read_next_batch():
            table = pa.Table.from_batches([batch])
            df = pl.from_arrow(table)

            for row in df.iter_rows(named=True):
                if max_records and total_loaded >= max_records:
                    break
                split = 'train' #row["set"]
                if isinstance(split, dict):
                    split = list(split.values())[0]  # handle dictionary field
                question = f"What is the IUPAC name for the molecule {row['smiles']}?"
                answer = f"It is {row['iupac']}"
                full_text = f"{question} {answer}"
                data[split].append({"text": full_text})
                total_loaded += 1

            if max_records and total_loaded >= max_records:
                break

        dataset = DatasetDict({
            split: Dataset.from_list(samples)
            for split, samples in data.items() if samples
        })
        return dataset

def main():
    parser = argparse.ArgumentParser(description="Train Phi-4 on IUPAC chemical naming task.")
    parser.add_argument("--arrow_file", type=str, default='/home/lyg/source/l1nus/etl/pubchem/pubchem/pubchem.arrow', help="Path to the Arrow file.")
    parser.add_argument("--output_dir", type=str, default="./phi4-iupac-model")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=4096
    parser.add_argument("--max_records", type=int, default=1000, help="Limit the number of records loaded.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    logger.info("Loading and preparing dataset...")
    dataset = load_and_prepare_data(args.arrow_file, args.max_records)

    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-4", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-4", trust_remote_code=True)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=args.max_length, padding="max_length"
        )

    logger.info("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output_dir, "logs"),
        save_strategy="epoch",
        push_to_hub=False,
        logging_steps=10,
        report_to="none"
    )

    logger.info("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets.get("train"),
        eval_dataset=tokenized_datasets.get("valid"),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Training complete and model saved.")

if __name__ == "__main__":
    main()