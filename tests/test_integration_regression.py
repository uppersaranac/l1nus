"""Integration regression test: new pipeline vs legacy expansion.

Compares the **tokenised** outputs of the old expansion+tokenisation flow
(`QuestionSetProcessor.expand_dataset` → `process_single_qa`) against the new
pipeline (QuestionGenerator + preprocess helpers) on a small synthetic dataset.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
import json
import yaml
import pandas as pd
import pytest
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from llm.llm_apis import QUESTION_SETS, QuestionSetProcessor, process_single_qa
from llm.questions.generators import GenerationConfig, QuestionGenerator
from llm.datasets.preprocess import tokenise_split, split_by_column


@pytest.mark.integration
def test_pipeline_parity(tmp_path: Path):
    # --------------------------
    # 1. Build raw dataset (2 molecules)
    # --------------------------
    smiles = ["C", "CC"]
    df_raw = pd.DataFrame({"smiles": smiles})

    # --------------------------
    # 2. OLD pipeline – expand via QuestionSetProcessor
    # --------------------------
    set_name = "iupac_naming"
    proc_cls = QuestionSetProcessor  # base, but we instantiate subclass via registry below
    # Dynamically fetch processor class implemented elsewhere
    from llm.questions.processors import PROCESSOR_CLASSES

    proc = PROCESSOR_CLASSES[set_name]()
    answers_dict = proc.prepare_answers({"smiles": smiles})
    old_expanded = proc.expand_dataset({"smiles": smiles}, answers_dict)

    # Old expanded returns HF Dataset already
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    old_tok = tokenise_split(old_expanded, tokenizer, is_train=True)

    # --------------------------
    # 3. NEW pipeline – QuestionGenerator path
    # --------------------------
    # Create YAML config on the fly based on QUESTION_SETS
    cfg_data = {
        "system_prompt": QUESTION_SETS[set_name]["system_prompt"],
        "questions": QUESTION_SETS[set_name]["questions"],
    }
    yaml_path = tmp_path / "qcfg.yaml"
    yaml_path.write_text(yaml.dump(cfg_data))

    # Add answer columns needed by generator
    for q in cfg_data["questions"]:
        qid = q["id"]
        df_raw[qid] = answers_dict[qid]
    df_raw["split"] = "train"

    gen_cfg = GenerationConfig.from_yaml(yaml_path)
    generator = QuestionGenerator(gen_cfg)

    qa_dicts = []
    for qa in generator.generate(df_raw):
        qid = qa["question_id"]
        tmpl_dict = next(q for q in cfg_data["questions"] if q["id"] == qid)
        qa_dicts.append(
            {
                "smiles": qa["metadata"]["smiles"],
                "question_id": qid,
                "question_template": tmpl_dict["user_template"],
                "answer": answers_dict[qid][smiles.index(qa["metadata"]["smiles"])],
                "assistant_template": tmpl_dict["assistant_template"],
                "system_prompt": cfg_data["system_prompt"],
            }
        )

    new_ds = Dataset.from_list(qa_dicts)
    new_tok = tokenise_split(new_ds, tokenizer, is_train=True)

    # --------------------------
    # 4. Compare tokenised outputs – order may differ; compare sets
    # --------------------------
    old_tuples = {(tuple(x["input_ids"]), tuple(x["labels"])) for x in old_tok}
    new_tuples = {(tuple(x["input_ids"]), tuple(x["labels"])) for x in new_tok}

    assert old_tuples == new_tuples, "Tokenised outputs diverged between old and new pipeline"
