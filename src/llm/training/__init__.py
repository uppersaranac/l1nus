"""Training sub-package.

Currently provides a thin wrapper (`cli_train.py`) around HuggingFace Trainer
so that we can fine-tune a causal-LM using datasets prepared by
`llm.datasets.cli_build`.

Future work: extract / reuse GenTrainer & callbacks from the legacy
`train_llm.py` for full parity.  For now we keep a minimal variant that calls
Seq2SeqTrainer with the existing `compute_metrics_closure` from `llm_apis`.
"""
