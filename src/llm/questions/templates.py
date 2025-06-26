"""Generic question template definitions.

These classes are intentionally simple and domain-agnostic. They only know
about *string templating* – they do **not** know how to calculate an answer for
any particular domain (chemistry, maths, etc.).

A template record comes from YAML and contains at minimum:
    id: unique name
    type: mcq | fill_blank | freeform | generic
    user_template: e.g. "How many carbons in {smiles}?"
    assistant_template: e.g. "<|extra_100|>{answer}<|extra_101|>"

Additional fields are kept untouched and passed through to the rendered
question dictionary as-is.

Usage::

    tmpl = QuestionTemplate(**yaml_dict)
    q_str = tmpl.render_question({"smiles": "CCO"})
    a_str = tmpl.render_answer({"answer": 3})
"""
from __future__ import annotations

from dataclasses import dataclass, field
from string import Template
from typing import Dict, Any

__all__ = [
    "QuestionTemplate",
    "template_from_dict",
]

@dataclass
class QuestionTemplate:
    """
    A lightweight question/answer template wrapper.

    :param id: Unique template identifier.
    :type id: str
    :param user_template: User-facing question template string.
    :type user_template: str
    :param assistant_template: Assistant answer template string.
    :type assistant_template: str
    :param type: Template type (e.g., mcq, fill_blank, freeform, etc.).
    :type type: str
    :param extras: Arbitrary extra fields from YAML.
    :type extras: Dict[str, Any]
    """

    id: str
    user_template: str
    assistant_template: str
    type: str = "generic"  # mcq | fill_blank | freeform | etc.
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QuestionTemplate":
        """
        Create a QuestionTemplate instance from a dictionary.

        :param d: Dictionary with template fields.
        :type d: Dict[str, Any]
        :return: QuestionTemplate instance.
        :rtype: QuestionTemplate
        """
        known_keys = {"id", "user_template", "assistant_template", "type"}
        extras = {k: v for k, v in d.items() if k not in known_keys}
        return cls(
            id=d["id"],
            user_template=d["user_template"],
            assistant_template=d["assistant_template"],
            type=d.get("type", "generic"),
            extras=extras,
        )

def template_from_dict(d: Dict[str, Any]) -> QuestionTemplate:
    """
    Factory so external code can stay generic in the future.

    :param d: Dictionary with template fields.
    :type d: Dict[str, Any]
    :return: QuestionTemplate instance.
    :rtype: QuestionTemplate
    """
    return QuestionTemplate.from_dict(d)
