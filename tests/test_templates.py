"""Unit tests for question template rendering."""
import pytest

from llm.questions.templates import QuestionTemplate


def test_render_question_and_answer():
    tmpl_dict = {
        "id": "count_carbons",
        "type": "freeform",
        "user_template": "How many carbons in {smiles}?",
        "assistant_template": "The answer is {answer}.",
    }
    tmpl = QuestionTemplate.from_dict(tmpl_dict)

    mapping = {"smiles": "CCO", "answer": 2}
    q = tmpl.render_question(mapping)
    a = tmpl.render_answer(mapping)

    assert q == "How many carbons in CCO?"
    assert a == "The answer is 2."
