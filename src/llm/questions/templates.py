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


def _safe_render(tmpl: str, mapping: Dict[str, Any]) -> str:
    """Render *tmpl* with *mapping* using str.format style placeholders.

    If a key is missing in *mapping*, a KeyError is raised so that the caller
    can decide whether to ignore or surface the problem.
    """
    return tmpl.format(**mapping)


@dataclass
class QuestionTemplate:
    """A lightweight question/answer template wrapper."""

    id: str
    user_template: str
    assistant_template: str
    type: str = "generic"  # mcq | fill_blank | freeform | etc.
    # We allow arbitrary extra fields from YAML
    extras: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def render_question(self, mapping: Dict[str, Any]) -> str:
        return _safe_render(self.user_template, mapping)

    def render_answer(self, mapping: Dict[str, Any]) -> str:
        return _safe_render(self.assistant_template, mapping)

    # ------------------------------------------------------------------
    # YAML helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QuestionTemplate":
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
    """Factory so external code can stay generic in the future."""
    return QuestionTemplate.from_dict(d)
