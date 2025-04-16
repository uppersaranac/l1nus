
#!/usr/bin/env python
import argparse
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments


"""
property_schema_fields = [
    pa.field("cid", pa.uint64()),
    pa.field("complexity", pa.float32()),
    pa.field("hba", pa.int32()),
    pa.field("hbd", pa.int32()),
    pa.field("rotatable_bonds", pa.int32()),
    pa.field("tpsa", pa.float32()),
    pa.field("logp", pa.float32()),
    pa.field("monoisotopic_mass", pa.float64()),
    pa.field("exact_mass", pa.float64()),
    pa.field("formula", pa.string()),
    pa.field("molecular_weight", pa.float64()),
    pa.field("charge", pa.int32()),
    pa.field("num_atoms", pa.int32()),
    pa.field("num_def_stereo", pa.int32()),
    pa.field("num_undef_stereo", pa.int32()),
    pa.field("num_def_double", pa.int32()),
    pa.field("num_undef_double", pa.int32()),
    pa.field("num_isotopic_atoms", pa.int32()),
    pa.field("fragments", pa.int32()),
    pa.field("num_tautomers", pa.int32()),
    pa.field("num_complexity", pa.int32()),
    pa.field("iupac_openeye", pa.string()),
    pa.field("iupac_cas", pa.string()),
    pa.field("iupac", pa.string()),
    pa.field("iupac_systematic", pa.string()),
    pa.field("iupac_traditional", pa.string()),
    pa.field("smiles", pa.string()),
    pa.field("set", pa.dictionary(pa.int32(), pa.string()))
]

schema = pa.schema(property_schema_fields)

There are 3 files for training, validation, and test data. Read and process the datasets in batch as the
 datasets are very large. We will be fine tuning instruction trained models like Qwen/Qwen2.5-0.5B-Instruct.  
 The training task should be of question and answer format where the question is of the format  
 f"What is the IUPAC name for the molecule {smiles}?" and the answer is of the format "It is {iupac}" 
 where {smiles} corresponds to the 'smiles' field in arrow schema and {iupac} comes from the 'iupac' 
 field in the arrow schema. The code to create the questions and answers should not be hardcoded for which 
 model is selected and use special tokens extracted from the model and tokenizer configuration.
use tokenizer.apply_chat_template() instead of hard coding the format of the question and answer. 
Please add logging, accuracy and perplexity metrics, command line arguments for the filename of the data and applicable TrainingArguments, 
and add an argument to limit the number of records loaded for training. Instead of using dataframes,
 just load the Datasets from the arrow files directly. Use a batched Dataset.map() to convert the 
 data into tokenized datasets for training and evaluation. It's important to minimize memory usage
   by batch processing in this script. 

"""

def load_arrow_dataset(file_path, max_records=None):
    """
    Load an Arrow file using PyArrow, optionally limiting the number of records.
    The resulting Arrow table is then converted into a Hugging Face Dataset.
    """
    logging.info(f"Loading dataset from {file_path}")
    # try:
    #     with pa.memory_map(file_path, 'rb') as source:
    #         table = pa.ipc.read_table(source)
    # except Exception as e:
    #     logging.error(f"Error loading {file_path}: {e}")
    #     sys.exit(1)
    # if max_records is not None:
    #     table = table.slice(0, max_records)
    # return Dataset.from_arrow(table)
    dataset = Dataset.from_file(str(Path(file_path).expanduser()))
    if max_records is not None and len(dataset) > max_records:
        dataset = dataset.select(range(max_records))
    return dataset


def compute_metrics(eval_preds):
    """
    Compute token-level accuracy and perplexity.
   
    Accuracy is calculated by comparing the argmax predictions vs. labels
    (ignoring pad tokens marked as -100), and perplexity is exp(cross_entropy_loss).

    output is 
    eval_preds.label_ids of type numpy.ndarray and shape 237, 512 and dtype int64
    eval_preds.predictions of type numpy.dtypes.Float32DType  shape (237, 512, 151936) and dtype float32

    """
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    mask = labels != -100
    if mask.sum() > 0:
        accuracy = (predictions[mask] == labels[mask]).mean()
    else:
        accuracy = 0.0

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    logits_tensor = torch.tensor(logits)
    labels_tensor = torch.tensor(labels)
    loss = loss_fct(
        logits_tensor.view(-1, logits_tensor.size(-1)),
        labels_tensor.view(-1)
    ).item()
    perplexity = math.exp(loss) if loss < 100 else float("inf")
    return {"accuracy": accuracy, "perplexity": perplexity}

# ---------------------------------------------------------------------
# Pretty‑print helper
# ---------------------------------------------------------------------
def print_predictions(trainer: Trainer, dataset: Dataset, tokenizer, title: str):
    """
    Runs `trainer.predict()` on *dataset*, decodes predictions & labels,
    and prints them side‑by‑side.
    """
    logging.info("Decoding %s set predictions …", title)
    output = trainer.predict(dataset)
    pred_ids = np.argmax(output.predictions, axis=-1)

    # Replace -100 in labels with pad token so we can decode cleanly
    pad_id = tokenizer.pad_token_id
    label_ids = np.where(output.label_ids == -100, pad_id, output.label_ids)

    predictions = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    references  = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    for i, (ref, pred) in enumerate(zip(references, predictions)):
        print(f"\n—— {title} example {i} ——")
        print("GROUND TRUTH:", ref.strip())
        print("PREDICTED   :", pred.strip())


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune an instruction-tuned model (e.g., microsoft/phi-4) on a Q&A task using Hugging Face Transformers"
    )
    parser.add_argument("--train_file", type=str, default='~/data/pubchem/arrow/test_train.arrow', help="Path to the training Arrow file.")
    parser.add_argument("--eval_file", type=str, default='~/data/pubchem/arrow/test_valid.arrow', help="Path to the evaluation Arrow file.")
    parser.add_argument("--test_file", type=str, default=None, help="Path to the test Arrow file (optional).")
    parser.add_argument("--output_dir", type=str, default="~/results", help="Path to the output directory.")
    parser.add_argument("--max_records", type=int, default=1000, help="Limit number of records loaded for training.")
    parser.add_argument("--eval_limit",  type=int, default=50)
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length for tokenization.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Training batch size per device.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Evaluation batch size per device.")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X update steps.")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Run an evaluation every X steps.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Pre-trained model name or path (e.g., microsoft/phi-4, Qwen/Qwen2.5-0.5B-Instruct).")
    args = parser.parse_args()

    # Setup logging.
    logging.basicConfig(level=logging.INFO)

    logging.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # Load datasets directly from Arrow files.
    logging.info("Loading the training dataset from Arrow file...")
    train_dataset = load_arrow_dataset(args.train_file, args.max_records)
    logging.info("Loading the evaluation dataset from Arrow file...")
    eval_dataset = load_arrow_dataset(args.eval_file, args.max_records)
    test_dataset = load_arrow_dataset(args.test_file, args.max_records) if args.test_file else None

    if args.eval_limit and len(eval_dataset) > args.eval_limit:
        eval_dataset = eval_dataset.select(range(args.eval_limit))

    def tokenize_batch(examples):
        """
        For a batch of examples, create a chat-format prompt and tokenize it.
        Each example creates a conversation where the user asks for the IUPAC name
        given the molecule SMILES and the assistant provides the answer.
        """
        texts = []
        for smiles, iupac in zip(examples["smiles"], examples["iupac"]):
            messages = [
                {"role": "system", "content": "You are a helpful chemistry professor."},
                {"role": "user", "content": f"What is the IUPAC name for the molecule {smiles}?"},
                {"role": "assistant", "content": f"It is {iupac}"}
            ]
            texts.append(tokenizer.apply_chat_template(conversation=messages, padding="max_length", truncation=True, max_length=args.max_length))
        return {'input_ids': texts}
#        return tokenizer(texts, padding="max_length", truncation=True, max_length=args.max_length)

    # Tokenize the datasets using batched mapping.
    logging.info("Tokenizing the training dataset in batches...")
    train_dataset = train_dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=1000,
        remove_columns=["smiles", "iupac"]
    )
    train_dataset = train_dataset.add_column('labels', train_dataset['input_ids'].copy())
    logging.info("Tokenizing the evaluation dataset in batches...")
    eval_dataset = eval_dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=1000,
        remove_columns=["smiles", "iupac"]
    )
    eval_dataset = eval_dataset.add_column('labels', eval_dataset['input_ids'].copy())

    if test_dataset is not None:
        logging.info("Tokenizing the test dataset in batches...")
        test_dataset = test_dataset.map(
            tokenize_batch,
            batched=True,
            batch_size=1000,
            remove_columns=["smiles", "iupac"]
        )
        test_dataset = test_dataset.add_column('labels', test_dataset['input_ids'].copy())

    # Set up TrainingArguments.
    training_args = TrainingArguments(
        output_dir=str(Path(args.output_dir).expanduser()),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=2,
        logging_dir=os.path.join(args.output_dir, "logs"),
        # load_best_model_at_end=True,
        metric_for_best_model="perplexity",
        eval_accumulation_steps=1,
    )

    # Initialize the Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Train and evaluate.
    logging.info("Starting training...")
    trainer.train()
    logging.info("Training complete. Evaluating the model on the evaluation dataset...")
    eval_results = trainer.evaluate()
    logging.info(f"Evaluation Results: {eval_results}")
    print_predictions(trainer, eval_dataset, tokenizer, "validation")

    if test_dataset is not None:
        logging.info("Evaluating the model on the test dataset...")
        test_results = trainer.evaluate(test_dataset)
        logging.info(f"Test Results: {test_results}")
        print_predictions(trainer, test_dataset, tokenizer, "test")

    # Save the model.
    logging.info("Saving the final model...")
    trainer.save_model(args.output_dir)
    logging.info("All done.")


if __name__ == "__main__":
    main()
