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
    assert longest_chain_length(mol) == 2
    mol2 = Chem.MolFromSmiles("C1CCCCC1") # cyclohexane (no acyclic chain)
    assert longest_chain_length(mol2) == 0

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

def test_norm_without_tokenizer():
    """Verify that _norm extracts the value after the last colon and space without tokenizer."""
    assert _norm("IUPAC name: ethanol\n") == "ethanol"
    assert _norm("The molecule's name: methane ") == "methane"
    assert _norm("Carbon count: 5") == "5"
    assert _norm("Answer: benzene.") == "benzene"
    # Test fallback to last word when no colon
    assert _norm("The answer is ethanol") == "ethanol"
    assert _norm("Result: 5 atoms") == "5"  # Still uses colon if present
    assert _norm("Just benzene") == "benzene"  # No whitespace, whole string
    assert _norm("benzene.") == "benzene"  # Removes trailing period

def test_norm_with_tokenizer(tokenizer):
    """Verify that _norm properly handles EOS tokens from tokenizer."""
    eos_token = tokenizer.eos_token  # Should be "<|im_end|>" for Qwen
    
    # Test with colon and EOS token
    assert _norm(f"Carbon count: 5{eos_token}", tokenizer) == "5"
    assert _norm(f"Answer: benzene{eos_token}", tokenizer) == "benzene"
    assert _norm(f"IUPAC name: ethanol.{eos_token}", tokenizer) == "ethanol"
    
    # Test fallback to last word with EOS token
    assert _norm(f"The answer is ethanol{eos_token}", tokenizer) == "ethanol"
    assert _norm(f"Just benzene{eos_token}", tokenizer) == "benzene"
    
    # Test stopping at whitespace after colon
    assert _norm("Carbon count: 5 more text", tokenizer) == "5"
    
    # Test stopping at period after colon
    assert _norm("Answer: benzene. More text", tokenizer) == "benzene"

def test_norm_edge_cases(tokenizer):
    """Test edge cases for _norm function."""
    eos_token = tokenizer.eos_token
    
    # Empty and whitespace strings
    assert _norm("", tokenizer) == ""
    assert _norm("   ", tokenizer) == ""
    assert _norm(f"   {eos_token}", tokenizer) == ""
    
    # Colon at the end
    assert _norm("Answer: ", tokenizer) == ""
    assert _norm(f"Answer: {eos_token}", tokenizer) == ""
    
    # Multiple colons - should use the last one
    assert _norm("First: ignore Second: keep", tokenizer) == "keep"
    
    # EOS token in the middle (should be removed)
    assert _norm(f"Answer: 5{eos_token}extra", tokenizer) == "5"
    
    # No colon, multiple words
    assert _norm("This is the answer", tokenizer) == "answer"
    assert _norm(f"This is the answer{eos_token}", tokenizer) == "answer"

def test_norm_eos_token_priority():
    """Test that different EOS tokens are handled correctly."""
    # Mock tokenizer with different EOS tokens
    mock_tokenizer_qwen = MagicMock()
    mock_tokenizer_qwen.eos_token = "<|im_end|>"  # Correct Qwen EOS token
    
    mock_tokenizer_llama = MagicMock()
    mock_tokenizer_llama.eos_token = "</s>"
    
    # Test Qwen-style EOS token
    assert _norm("Answer: 5<|im_end|>", mock_tokenizer_qwen) == "5"
    assert _norm("Just answer<|im_end|>", mock_tokenizer_qwen) == "answer"
    
    # Test Llama-style EOS token
    assert _norm("Answer: 5</s>", mock_tokenizer_llama) == "5"
    assert _norm("Just answer</s>", mock_tokenizer_llama) == "answer"
    
    # Test with no tokenizer (should still work)
    assert _norm("Answer: 5<|im_end|>") == "5<|im_end|>"  # No tokenizer, keeps EOS
    assert _norm("Just answer") == "answer"

def test_norm_complex_patterns():
    """Test complex patterns that might occur in model outputs."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token = "<|im_end|>"  # Correct Qwen EOS token
    
    # Repetitive generation (the original problem)
    repetitive = "The number of sulfur atoms is: 0The number of sulfur atoms is: 0<|im_end|>"
    assert _norm(repetitive, mock_tokenizer) == "0"  # Gets only the number, stops at 'T'
    
    # Multiple periods
    assert _norm("Answer: value....", mock_tokenizer) == "value"
    
    # Mixed punctuation
    assert _norm("Result: 5; additional info", mock_tokenizer) == "5;"
    
    # Nested colons
    assert _norm("Time: 10:30, Count: 5", mock_tokenizer) == "5"
    
    # Scientific notation - should stop at period
    assert _norm("Concentration: 1.5e-6", mock_tokenizer) == "1"

def test_process_single_qa_eos_integration(tokenizer):
    """Simplified test to verify EOS token integration."""
    from llm.llm_apis import process_single_qa
    
    example = {
        "system_prompt": "You are a helpful assistant.",
        "question_template": "How many carbon atoms in {smiles}?",
        "assistant_template": "The answer is {carbon_count}",
        "metadata": {"smiles": "CCO", "carbon_count": "2"}
    }
    
    # Test evaluation mode (simpler, no complex mocking needed)
    result = process_single_qa(
        tok=tokenizer,
        example=example,
        max_len=50,
        max_label_len=20,
        is_train=False  # Evaluation mode is simpler
    )
    
    # Verify that the function completed successfully
    assert "input_ids" in result
    assert "attention_mask" in result
    assert "labels" in result
    
    # Verify EOS token was used in the label formatting
    assert len(result["labels"]) == 50  # max_len

def test_norm_realistic_model_outputs(tokenizer):
    """Test with realistic model outputs that might cause issues."""
    eos_token = tokenizer.eos_token  # Should be "<|im_end|>" for Qwen
    
    # Typical good response
    good_response = f"The number of carbon atoms is: 6{eos_token}"
    assert _norm(good_response, tokenizer) == "6"
    
    # Response with explanation
    explained_response = f"The number of carbon atoms is: 6. This molecule is benzene.{eos_token}"
    assert _norm(explained_response, tokenizer) == "6"
    
    # Natural language response without colon
    natural_response = f"This molecule contains 6 carbon atoms{eos_token}"
    assert _norm(natural_response, tokenizer) == "atoms"
    
    # Response with number at the end
    natural_number = f"The answer is 6{eos_token}"
    assert _norm(natural_number, tokenizer) == "6"
    
    # Response with period at the end
    period_response = f"The count is 6.{eos_token}"
    assert _norm(period_response, tokenizer) == "6"

def test_process_single_qa_integration():
    """Test that process_single_qa correctly uses EOS tokens from tokenizer."""
    # This is an integration test to make sure the EOS token flows through correctly
    from llm.llm_apis import process_single_qa
    
    # Create a simple tokenizer mock
    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token = "<|endoftext|>"
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.return_value = {
        "input_ids": np.array([[1, 2, 3, 0, 0]]),  # Mock tokenized input
        "attention_mask": np.array([[1, 1, 1, 0, 0]])
    }
    
    example = {
        "system_prompt": "You are a helpful assistant.",
        "question_template": "How many carbon atoms in {smiles}?",
        "assistant_template": "The number of carbon atoms is: {carbon_count}",
        "metadata": {"smiles": "CCO", "carbon_count": "2"}
    }
    
    # Test training mode
    result = process_single_qa(
        tok=mock_tokenizer,
        example=example,
        max_len=50,
        is_train=True
    )
    
    # Verify that the function was called and EOS token was used
    assert "input_ids" in result
    assert "attention_mask" in result
    assert "labels" in result
    
    # Check that the tokenizer was called with EOS token in the prompt
    mock_tokenizer.assert_called()
    call_args = mock_tokenizer.call_args[0][0]  # First positional argument
    assert "<|endoftext|>" in call_args  # EOS token should be in the prompt

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