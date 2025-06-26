"""Generic question generation utilities.

This module provides a **domain-agnostic** generator that expands raw tabular
records into one JSON-serialisable dict per question/answer pair.

The behaviour is defined entirely by a YAML configuration file – see
`cli_generate.py --help` for usage.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pyarrow as pa
import yaml

from .templates import QuestionTemplate, template_from_dict

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """
    Parsed YAML configuration container.

    :param system_prompt: System prompt string.
    :type system_prompt: str
    :param question_set: Name of the question set.
    :type question_set: str
    :param question_templates: List of QuestionTemplate instances.
    :type question_templates: List[QuestionTemplate]
    """

    system_prompt: str
    question_set: str
    question_templates: List[QuestionTemplate]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GenerationConfig":
        """
        Create a GenerationConfig from a YAML file.

        :param path: Path to YAML configuration file.
        :type path: str or Path
        :return: GenerationConfig instance.
        :rtype: GenerationConfig
        """
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        q_tmpls = [template_from_dict(d) for d in cfg["questions"]]
        return cls(system_prompt=cfg["system_prompt"], question_set=cfg['question_set'], question_templates=q_tmpls)


class QuestionGenerator:
    """
    Expand raw records into question/answer pairs using *GenerationConfig*.

    :param config: GenerationConfig instance.
    :type config: GenerationConfig
    """

    def __init__(self, config: GenerationConfig):
        """
        Initialize the QuestionGenerator.

        :param config: GenerationConfig instance.
        :type config: GenerationConfig
        """
        self.config = config

    def generate(self, table: pa.Table) -> Iterator[Dict[str, Any]]:
        """
        Yield a dict per Q-A pair from an Arrow Table.

        Each output dict contains:
            question            rendered user prompt
            answer              rendered assistant answer
            question_id         template id
            assistant_template  original assistant template string
            metadata            original record as dict (caller can drop large cols)

        :param table: Arrow Table containing data.
        :type table: pa.Table
        :yield: Dictionary for each Q-A pair.
        :rtype: Iterator[Dict[str, Any]]
        """
        for i in range(table.num_rows):
            record = {k: v[0] for k, v in table.slice(i, 1).to_pydict().items()}
            for q_tmpl in self.config.question_templates:
                metadata = {k: v for k, v in record.items() if k != "split"}
                yield {
                    "question_id": q_tmpl.id,
                    "question_template": q_tmpl.user_template,
                    "assistant_template": q_tmpl.assistant_template,
                    "split": record.get("split", "train"),
                    "metadata": metadata,
                }

    def generate_jsonl(self, table: pa.Table, out_path: str | Path) -> int:
        """
        Write JSONL file from an Arrow Table. Returns number of examples written.

        :param table: Arrow Table containing data.
        :type table: pa.Table
        :param out_path: Output file path.
        :type out_path: str or Path
        :return: Number of examples written.
        :rtype: int
        """
        n = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for qa in self.generate(table):
                json.dump(qa, f, ensure_ascii=False)
                f.write("\n")
                n += 1
        return n
