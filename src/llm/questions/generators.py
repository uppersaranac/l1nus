"""Generic question generation utilities.

This module provides a **domain-agnostic** generator that expands raw tabular
records into one JSON-serialisable dict per question/answer pair.

The behaviour is defined entirely by a YAML configuration file – see
`cli_generate.py --help` for usage.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Iterator, Sequence

import yaml
import pandas as pd

from .templates import QuestionTemplate, template_from_dict

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Parsed YAML configuration container."""

    system_prompt: str
    question_templates: List[QuestionTemplate]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GenerationConfig":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        q_tmpls = [template_from_dict(d) for d in cfg["questions"]]
        return cls(system_prompt=cfg["system_prompt"], question_templates=q_tmpls)


class QuestionGenerator:
    """Expand raw records into question/answer pairs using *GenerationConfig*."""

    def __init__(self, config: GenerationConfig):
        self.config = config

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------
    def generate(self, df: pd.DataFrame) -> Iterator[Dict[str, Any]]:
        """Yield a dict per Q-A pair.

        Each output dict contains:
            question      – rendered user prompt
            answer        – rendered assistant answer
            question_id   – template id
            assistant_template – original assistant template string
            system_prompt – from config.system_prompt
            metadata      – original record as dict (caller can drop large cols)
        """
        for _, row in df.iterrows():
            record = row.to_dict()
            for q_tmpl in self.config.question_templates:
                mapping = record | {"answer": record.get(q_tmpl.id)}  # answer_column default to id
                try:
                    question_text = q_tmpl.render_question(record)
                    answer_text = q_tmpl.render_answer(mapping)
                except KeyError as exc:
                    logger.debug("Skipping template %s due to missing key: %s", q_tmpl.id, exc)
                    continue

                yield {
                    "question": question_text,
                    "answer": answer_text,
                    "question_id": q_tmpl.id,
                    "assistant_template": q_tmpl.assistant_template,
                    "system_prompt": self.config.system_prompt,
                    "metadata": record,
                    "split": record.get("split", "train"),
                }

    # ------------------------------------------------------------------
    # CONVENIENCE HELPERS
    # ------------------------------------------------------------------
    def generate_jsonl(self, df: pd.DataFrame, out_path: str | Path) -> int:
        """Write JSONL file. Returns number of examples written."""
        n = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for qa in self.generate(df):
                json.dump(qa, f, ensure_ascii=False)
                f.write("\n")
                n += 1
        return n
