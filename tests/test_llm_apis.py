import pytest
import numpy as np
import torch
import pyarrow as pa

from unittest.mock import MagicMock
from transformers import AutoTokenizer
from llm.llm_apis import (
    _norm, compute_metrics_closure, do_generation,
    count_heavy_atoms, count_non_hydrogen_bonds, count_positive_formal_charge_atoms, count_negative_formal_charge_atoms,
    calculate_molecular_properties,
    QuestionSetProcessor, IUPACNamingProcessor, MolecularPropertiesProcessor, AllPropertiesProcessor,

)

@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token
    return tok

def test_norm():
    """Verify that _norm extracts the value between the special markers."""
    assert _norm("It is <|extra_100|>ethanol<|extra_101|>") == "ethanol"
    assert _norm(" <|extra_100|>ethanol<|extra_101|> ") == "ethanol"

def test_count_heavy_atoms():
    assert count_heavy_atoms("CCO") == 3  # ethanol: 2C, 1O
    assert count_heavy_atoms("[NH4+]") == 1
    assert count_heavy_atoms("") == 0

def test_count_non_hydrogen_bonds():
    assert count_non_hydrogen_bonds("CCO") == 2  # C-C and C-O
    assert count_non_hydrogen_bonds("C") == 0
    assert count_non_hydrogen_bonds("") == 0

def test_count_positive_formal_charge_atoms():
    assert count_positive_formal_charge_atoms("[NH4+]") == 1
    assert count_positive_formal_charge_atoms("CCO") == 0
    assert count_positive_formal_charge_atoms("") == 0

def test_count_negative_formal_charge_atoms():
    assert count_negative_formal_charge_atoms("[O-][N+](=O)O") == 1  # nitrite anion
    assert count_negative_formal_charge_atoms("CCO") == 0
    assert count_negative_formal_charge_atoms("") == 0

def test_calculate_molecular_properties():
    smiles = ["CCO", "[NH4+]"]
    props = calculate_molecular_properties(smiles)
    assert props["heavy_atom_count"] == [3, 1]
    assert props["non_hydrogen_bond_count"] == [2, 0]
    assert props["positive_formal_charge_count"] == [0, 1]
    assert props["negative_formal_charge_count"] == [0, 0]
    assert props["carbon_count"] == [2, 0]
    assert props["oxygen_count"] == [1, 0]

def test_iupac_naming_processor():
    # Create a table with smiles and iupac columns
    smiles = ["CCO", "CC(=O)O"]
    iupac = ["ethanol", "acetic acid"]
    table = pa.table({"smiles": smiles, "iupac": iupac})
    proc = IUPACNamingProcessor()
    answers = proc.prepare_answers(table)
    assert answers == {"iupac_name": ["ethanol", "acetic acid"]}

def test_molecular_properties_processor():
    smiles = ["CCO", "CC(=O)O"]
    iupac = ["ethanol", "acetic acid"]
    table = pa.table({"smiles": smiles, "iupac": iupac})
    proc = MolecularPropertiesProcessor()
    answers = proc.prepare_answers(table)
    # Check a few key properties for both molecules
    assert answers["carbon_count"] == ["2", "2"]
    assert answers["oxygen_count"] == ["1", "2"]
    assert answers["iupac_name"] == ["ethanol", "acetic acid"]

def test_all_properties_processor():
    smiles = ["CCO", "CC(=O)O"]
    iupac = ["ethanol", "acetic acid"]
    table = pa.table({"smiles": smiles, "iupac": iupac})
    proc = AllPropertiesProcessor()
    answers = proc.prepare_answers(table)
    assert answers["carbon_count"] == ["2", "2"]
    assert answers["iupac_name"] == ["ethanol", "acetic acid"]
    # Check other properties exist
    assert "heavy_atom_count" in answers
    assert "non_hydrogen_bond_count" in answers

def test_processors_edge_cases():
    # Edge case 1: Empty input
    smiles = []
    iupac = []
    table = pa.table({"smiles": smiles, "iupac": iupac})
    for Processor in [IUPACNamingProcessor, MolecularPropertiesProcessor, AllPropertiesProcessor]:
        proc = Processor()
        answers = proc.prepare_answers(table)
        # All returned lists should be empty
        for v in answers.values():
            assert v == []
    # Edge case 2: Invalid SMILES
    smiles = ["", "not_a_smiles"]
    iupac = ["", "invalid"]
    table = pa.table({"smiles": smiles, "iupac": iupac})
    for Processor in [MolecularPropertiesProcessor, AllPropertiesProcessor]:
        proc = Processor()
        answers = proc.prepare_answers(table)
        for v in answers.values():
            assert len(v) == 2
    proc = IUPACNamingProcessor()
    answers = proc.prepare_answers(table)
    assert answers["iupac_name"] == ["", "invalid"]
    # Edge case 3: Mismatched lengths
    smiles = ["CCO"]
    iupac = ["ethanol", "extra"]
    with pytest.raises(Exception):
        table = pa.table({"smiles": smiles, "iupac": iupac})
    for Processor in [IUPACNamingProcessor, MolecularPropertiesProcessor, AllPropertiesProcessor]:
        proc = Processor()
        try:
            answers = proc.prepare_answers(table)
        except Exception as e:
            assert isinstance(e, (IndexError, ValueError, AssertionError))

def test_question_set_processor_not_implemented():
    class Dummy(QuestionSetProcessor):
        pass
    dummy = Dummy("iupac_naming")
    table = pa.table({"smiles": [], "iupac": []})
    with pytest.raises(NotImplementedError):
        dummy.prepare_answers(table)

def test_compute_metrics_closure_exact_match(tokenizer):
    compute_metrics = compute_metrics_closure(tokenizer)
    ids = tokenizer("ethanol", add_special_tokens=False)["input_ids"]
    pred = np.array([ids])
    label = np.array([[i if i != tokenizer.pad_token_id else -100 for i in ids]])
    result = compute_metrics((pred, label))
    assert "exact_match" in result
    assert result["exact_match"] == 1.0

def test_do_generation_mock(tokenizer):
    # Tokenize two simple mock prompts
    texts = ["CCO", "CC(=O)O"]
    encodings = tokenizer(texts, padding='max_length', max_length=10, return_tensors='pt')
    dummy_input_ids = encodings['input_ids']
    dummy_attention_mask = encodings['attention_mask']

    class DummyDataset:
        def set_format(self, type=None, columns=None):
            pass
        def __getitem__(self, key):
            if key == "input_ids":
                return dummy_input_ids
            elif key == "attention_mask":
                return dummy_attention_mask

    model = MagicMock()
    model.device = torch.device("cpu")
    model.generate = MagicMock(return_value=torch.tensor([[1, 2, 3], [4, 5, 6]]))

    preds = do_generation(max_new_tokens=5,
                          tokenizer=tokenizer, model=model,
                          data=DummyDataset())
    assert isinstance(preds, list)
    assert all(isinstance(p, str) for p in preds)

    model.generate.assert_called_once()