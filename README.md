# l1nus

LLM fine-tuning toolkit for chemistry Q&A. The codebase turns molecular tables into templated question/answer pairs, tokenises them with Hugging Face tooling, trains causal LMs, evaluates exact-match accuracy, and ships a simple chat-style inference loop.

## Repo layout
- `configs/`: YAML templates defining system prompts and questions (e.g. molecular_properties, iupac_naming, structure).
- `src/llm/questions/`: Question generation CLI (`cli_generate.py`) plus template and processor plumbing.
- `src/llm/datasets/`: Dataset builder that tokenises generated Q&A into HF `DatasetDict` objects.
- `src/llm/training/`: Fine-tuning entrypoint (`cli_train.py`) built on Accelerate.
- `src/llm/evaluate/`: Evaluation CLI producing exact-match metrics and optional CSV dumps.
- `src/llm/inference/`: Lightweight chat loop for local checkpoints.
- `src/llm/llm_apis.py`, `src/llm/llm_mol.py`, `src/llm/structure_processor.py`: Core chemistry helpers (RDKit-driven property extraction, structure MCQ generation, prompt/label handling).
- `src/etl/mol/`: Scripts to stage raw chemistry datasets (PubChem subsets, Pistachio reactions, etc.).
- `scripts/`: SLURM/Accelerate launchers for multi-node or single-node runs.

## Setup
1. Python 3.8+ environment with [uv](https://github.com/astral-sh/uv) installed.
2. Install project dependencies with `uv sync` (it will create `.venv` automatically if missing):
   ```bash
   uv sync
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   ```
   - Use `uv sync --python 3.11` (or a full path) if you want to pin the interpreter for the new env.
   - If you need a GPU-specific Torch wheel, install it after syncing using the command from https://pytorch.org (e.g., `uv pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124`).
3. Optional dev/test extras:
   ```bash
   uv sync --extra dev
   ```

## End-to-end pipeline
1) **Generate questions** from raw tabular data (Arrow/Parquet/CSV/TSV/JSONL). Config selects the question set and templates; processors auto-compute answers for chemistry sets.
   ```bash
   python -m llm.questions.cli_generate \
     --input data/raw_molecules.parquet \
     --config configs/molecular_properties.yaml \
     --output data/questions.jsonl \
     --limit 5000 \
     --filter-column num_atoms --filter-min 3 --filter-max 60
   ```
   - Question sets map to processors: `molecular_properties`, `all_properties`, `iupac_naming`, `structure` (see `llm.questions.processors.PROCESSOR_CLASSES`).
   - Raw columns required vary by set (e.g. `smiles` for property counting, `iupac` for naming, `formula` + `smiles` for structure MCQs).

2) **Tokenise** questions into HF datasets ready for training/eval.
   ```bash
   python -m llm.datasets.cli_build \
     --questions data/questions.jsonl \
     --tokenizer Qwen/Qwen2.5-1.5B \
     --config configs/molecular_properties.yaml \
     --output data/tokenised/molecular_properties \
     --max-length 2048 \
     --max-label-len 512 \
     --create-position-weights
   ```
   - Writes `full/` (no train split) and `minimal/` subdirs under the output path.
   - Uses `llm.llm_apis.process_single_qa` so prompts/labels match training behaviour.

3) **Train** a causal LM with Accelerate.
   ```bash
   python -m llm.training.cli_train \
     --dataset_dir data/tokenised/molecular_properties \
     --model_name Qwen/Qwen2.5-1.5B \
     --output_dir runs/molecular_properties \
     --num_train_epochs 3 \
     --per_device_train_batch_size 2 \
     --per_device_eval_batch_size 2 \
     --max_new_tokens 512 \
     --eval_steps 1000
   ```
   - Supports optional position-wise loss weighting (`--use_position_weighting`) and sampling knobs (`--temperature`, `--top_p`, `--top_k`).

4) **Evaluate** exact-match accuracy and export predictions.
   ```bash
   python -m llm.evaluate.cli_evaluate \
     --dataset_dir data/tokenised/molecular_properties/full \
     --model_name runs/molecular_properties/best_model \
     --split test \
     --output_csv runs/molecular_properties/test_predictions.csv \
     --max_new_tokens 512
   ```
   - Reuses generation helpers from `llm.llm_apis` to ensure parity with training.

5) **Chat/inference** with a trained model.
   ```bash
   python -m llm.inference.cli_inference \
     --model_name runs/molecular_properties/best_model \
     --system_prompt "Answer with <answer> tags." \
     --history \
     --max_new_tokens 256
   ```
   - `--thinking` toggles models that support explicit reasoning tokens (e.g. Qwen3).

## Configuration notes
- YAML files in `configs/` define `system_prompt`, `question_set`, and templated `questions` (`user_template` / `assistant_template`).
- Processors in `llm.questions.processors` can enrich raw tables with computed answers before templating (RDKit-derived properties, IUPAC names, structural isomer options).
- Adjust max lengths and sampling flags in the CLIs to match your target model.

## Data utilities
- `src/etl/mol/pubchem/*.py`, `src/etl/mol/reactions/load_pistachio.py`: helpers to ingest and deduplicate upstream chemistry sources into Arrow/Parquet for the generation step.
- Scripts under `scripts/` show how to launch multi-node Accelerate runs on SLURM with optional DeepSpeed configs.

## Testing
Run the small regression/unit suite:
```bash
python -m pytest
```
