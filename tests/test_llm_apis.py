import pytest
import numpy as np
import torch
from rdkit import Chem
import pyarrow as pa

from unittest.mock import MagicMock
from transformers import AutoTokenizer
from llm.llm_apis import (
    _norm, compute_metrics_closure, do_generation,
    count_heavy_atoms, count_non_hydrogen_bonds, count_positive_formal_charge_atoms, count_negative_formal_charge_atoms,
    count_element_atoms, count_carbon_atoms, count_nitrogen_atoms, count_oxygen_atoms, count_sulfur_atoms, count_phosphorus_atoms,
    count_chlorine_atoms, count_fluorine_atoms, count_rings, count_aromatic_rings, count_double_bonds, count_triple_bonds,
    count_stereo_double_bonds, count_stereocenters, count_five_membered_rings, count_aromatic_five_membered_rings,
    count_six_membered_rings, count_aromatic_six_membered_rings, longest_chain_length, count_total_hydrogens,
    count_fused_rings, count_aromatic_heterocycles, count_aromatic_carbocycles,
    count_saturated_heterocycles, count_saturated_carbocycles, count_aliphatic_heterocycles, count_aliphatic_carbocycles,
    calculate_molecular_properties,
    QuestionSetProcessor, IUPACNamingProcessor, MolecularPropertiesProcessor, AllPropertiesProcessor,

)

def test_count_element_atoms():
    mol = Chem.MolFromSmiles("CCO")
    assert count_element_atoms(mol, 'C') == 2
    assert count_element_atoms(mol, 'O') == 1
    assert count_element_atoms(mol, 'N') == 0

def test_count_carbon_atoms():
    mol = Chem.MolFromSmiles("CCO")
    assert count_carbon_atoms(mol) == 2

def test_count_nitrogen_atoms():
    mol = Chem.MolFromSmiles("CCN")
    assert count_nitrogen_atoms(mol) == 1

def test_count_oxygen_atoms():
    mol = Chem.MolFromSmiles("CCO")
    assert count_oxygen_atoms(mol) == 1

def test_count_sulfur_atoms():
    mol = Chem.MolFromSmiles("CCS")
    assert count_sulfur_atoms(mol) == 1

def test_count_phosphorus_atoms():
    mol = Chem.MolFromSmiles("CP")
    assert count_phosphorus_atoms(mol) == 1

def test_count_chlorine_atoms():
    mol = Chem.MolFromSmiles("CCCl")
    assert count_chlorine_atoms(mol) == 1

def test_count_fluorine_atoms():
    mol = Chem.MolFromSmiles("CCF")
    assert count_fluorine_atoms(mol) == 1

def test_count_rings():
    mol = Chem.MolFromSmiles("C1CCCCC1") # cyclohexane
    assert count_rings(mol) == 1

def test_count_aromatic_rings():
    mol = Chem.MolFromSmiles("c1ccccc1") # benzene
    assert count_aromatic_rings(mol) == 1

def test_count_double_bonds():
    mol = Chem.MolFromSmiles("C=CC=C")
    assert count_double_bonds(mol) == 2

def test_count_triple_bonds():
    mol = Chem.MolFromSmiles("C#CC#C")
    assert count_triple_bonds(mol) == 2

def test_count_stereo_double_bonds():
    mol = Chem.MolFromSmiles("C/C=C/C")
    assert count_stereo_double_bonds(mol) == 1

def test_count_stereocenters():
    mol = Chem.MolFromSmiles("C[C@H](O)F")
    assert count_stereocenters(mol) == 1

def test_count_five_membered_rings():
    mol = Chem.MolFromSmiles("C1CCCC1")
    assert count_five_membered_rings(mol) == 1

def test_count_aromatic_five_membered_rings():
    mol = Chem.MolFromSmiles("c1cc[nH]c1") # pyrrole
    assert count_aromatic_five_membered_rings(mol) == 1

def test_count_six_membered_rings():
    mol = Chem.MolFromSmiles("C1CCCCC1")
    assert count_six_membered_rings(mol) == 1

def test_count_aromatic_six_membered_rings():
    mol = Chem.MolFromSmiles("c1ccccc1")
    assert count_aromatic_six_membered_rings(mol) == 1

def test_longest_chain_length():
    mol = Chem.MolFromSmiles("CCO")
    assert longest_chain_length(mol) == 3
    mol2 = Chem.MolFromSmiles("C1CCCCC1") # cyclohexane (no acyclic chain)
    assert longest_chain_length(mol2) == 1

def test_count_total_hydrogens():
    mol = Chem.MolFromSmiles("CCO")
    assert count_total_hydrogens(mol) == 6

def test_count_fused_rings():
    mol_naphthalene = Chem.MolFromSmiles('c1ccc2ccccc2c1')
    assert count_fused_rings(mol_naphthalene) == 2
    mol_cyclohexane = Chem.MolFromSmiles('C1CCCCC1')
    assert count_fused_rings(mol_cyclohexane) == 0

def test_count_aromatic_heterocycles():
    mol = Chem.MolFromSmiles('c1ccncc1') # pyridine
    assert count_aromatic_heterocycles(mol) == 1

def test_count_aromatic_carbocycles():
    mol = Chem.MolFromSmiles('c1ccccc1') # benzene
    assert count_aromatic_carbocycles(mol) == 1

def test_count_saturated_heterocycles():
    mol = Chem.MolFromSmiles('C1CCNC1') # pyrrolidine
    assert count_saturated_heterocycles(mol) == 1

def test_count_saturated_carbocycles():
    mol = Chem.MolFromSmiles('C1CCCCC1') # cyclohexane
    assert count_saturated_carbocycles(mol) == 1

def test_count_aliphatic_heterocycles():
    mol = Chem.MolFromSmiles('C1CCNC1') # pyrrolidine
    assert count_aliphatic_heterocycles(mol) == 1

def test_count_aliphatic_carbocycles():
    mol = Chem.MolFromSmiles('C1CCCCC1') # cyclohexane
    assert count_aliphatic_carbocycles(mol) == 1


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token
    return tok

def test_norm():
    """Verify that _norm extracts the value after the last colon and space."""
    assert _norm("IUPAC name: ethanol\n") == "ethanol"
    assert _norm("The molecule's name: methane ") == "methane"
    assert _norm("Carbon count: 5") == "5"
    assert _norm("Answer: benzene.") == "benzene"
    assert _norm("ethanol") == "ethanol"  # No colon, returns original stripped

def test_count_heavy_atoms():
    assert count_heavy_atoms(Chem.MolFromSmiles("CCO")) == 3  # ethanol: 2C, 1O
    assert count_heavy_atoms(Chem.MolFromSmiles("[NH4+]")) == 1

def test_count_non_hydrogen_bonds():
    assert count_non_hydrogen_bonds(Chem.MolFromSmiles("CCO")) == 2  # C-C and C-O
    assert count_non_hydrogen_bonds(Chem.MolFromSmiles("C")) == 0

def test_count_positive_formal_charge_atoms():
    assert count_positive_formal_charge_atoms(Chem.MolFromSmiles("[NH4+]")) == 1
    assert count_positive_formal_charge_atoms(Chem.MolFromSmiles("CCO")) == 0

def test_count_negative_formal_charge_atoms():
    assert count_negative_formal_charge_atoms(Chem.MolFromSmiles("[O-][N+](=O)O")) == 1  # nitrite anion
    assert count_negative_formal_charge_atoms(Chem.MolFromSmiles("CCO")) == 0

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
    assert answers[0] == {"iupac_name": ["ethanol", "acetic acid"]}

def test_molecular_properties_processor():
    smiles = ["CCO", "CC(=O)O"]
    iupac = ["ethanol", "acetic acid"]
    table = pa.table({"smiles": smiles, "iupac": iupac})
    proc = MolecularPropertiesProcessor()
    answers = proc.prepare_answers(table)
    # Check a few key properties for both molecules
    assert answers[0]["carbon_count"] == ["2", "2"]
    assert answers[0]["oxygen_count"] == ["1", "2"]
    assert answers[0]["iupac_name"] == ["ethanol", "acetic acid"]

def test_all_properties_processor():
    smiles = ["CCO", "CC(=O)O"]
    iupac = ["ethanol", "acetic acid"]
    table = pa.table({"smiles": smiles, "iupac": iupac})
    proc = AllPropertiesProcessor()
    answers = proc.prepare_answers(table)
    assert answers[0]["carbon_count"] == ["2", "2"]
    assert answers[0]["iupac_name"] == ["ethanol", "acetic acid"]
    # Check other properties exist
    assert "heavy_atom_count" in answers[0]
    assert "non_hydrogen_bond_count" in answers[0]

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