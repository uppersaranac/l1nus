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
import pyarrow as pa

from .templates import QuestionTemplate, template_from_dict

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Parsed YAML configuration container."""

    system_prompt: str
    question_set: str
    question_templates: List[QuestionTemplate]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GenerationConfig":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        q_tmpls = [template_from_dict(d) for d in cfg["questions"]]
        return cls(system_prompt=cfg["system_prompt"], question_set=cfg['question_set'], question_templates=q_tmpls)


class QuestionGenerator:
    """Expand raw records into question/answer pairs using *GenerationConfig*."""

    def __init__(self, config: GenerationConfig):
        self.config = config

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------
    def generate(self, table: pa.Table) -> Iterator[Dict[str, Any]]:
        """Yield a dict per Q-A pair from an Arrow Table.

        Each output dict contains:
            question      – rendered user prompt
            answer        – rendered assistant answer
            question_id   – template id
            assistant_template – original assistant template string
            metadata      – original record as dict (caller can drop large cols)
        """
        # Iterate row-by-row without converting full table to Python objects
        for i in range(table.num_rows):
            # Slice returns a Table with 1 row; convert to plain dict
            record = {k: v[0] for k, v in table.slice(i, 1).to_pydict().items()}
            for q_tmpl in self.config.question_templates:
                # Resolve answer value automatically; allow custom mapping per template
                answer_val = record.get(q_tmpl.id)
                # Special-case common chemistry field names
                if answer_val is None and q_tmpl.id == "iupac_name":
                    answer_val = record.get("iupac")

                mapping = record | {"answer": answer_val}
                try:
                    question_text = q_tmpl.render_question(record)
                    answer_text = q_tmpl.render_answer(mapping)
                except KeyError as exc:
                    logger.debug("Skipping template %s due to missing key: %s", q_tmpl.id, exc)
                    continue

                # Exclude split from metadata & top-level output
                metadata = {k: v for k, v in record.items() if k != "split"}

                yield {
                    "question": question_text,
                    "answer": answer_text,
                    "question_id": q_tmpl.id,
                    "question_template": q_tmpl.user_template,
                    "assistant_template": q_tmpl.assistant_template,
                    "split": record.get("split", "train"),
                    "metadata": metadata,
                }

    # ------------------------------------------------------------------
    # CONVENIENCE HELPERS
    # ------------------------------------------------------------------
    def generate_jsonl(self, table: pa.Table, out_path: str | Path) -> int:
        """Write JSONL file from an Arrow Table. Returns number of examples written."""
        n = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for qa in self.generate(table):
                json.dump(qa, f, ensure_ascii=False)
                f.write("\n")
                n += 1
        return n
