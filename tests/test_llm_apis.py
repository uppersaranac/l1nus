import pytest
import numpy as np
from unittest.mock import MagicMock

import pyarrow as pa
from rdkit import Chem
from transformers import AutoTokenizer

from llm.llm_apis import (
    _norm, _norm_tagged,
    calculate_molecular_properties,
    IUPACNamingProcessor, MolecularPropertiesProcessor, AllPropertiesProcessor
)
from llm.llm_mol import (
    count_heavy_atoms, count_non_hydrogen_bonds, count_positive_formal_charge_atoms, count_negative_formal_charge_atoms
)


def test_sorted_rings_basic():
    from rdkit import Chem
    from src.llm.llm_mol import sorted_rings
    mol = Chem.MolFromSmiles('C1CCCCC1')
    rings = sorted_rings(mol)
    assert len(rings) == 1
    assert sorted(rings[0], reverse=True) == [5, 4, 3, 2, 1, 0]

def test_kekulized_smiles_basic():
    from rdkit import Chem
    from src.llm.llm_mol import kekulized_smiles
    mol = Chem.MolFromSmiles('c1ccccc1')
    kek = kekulized_smiles(mol)
    assert isinstance(kek, str)
    kek_map = kekulized_smiles(mol, atom_map=True)
    assert '[' in kek_map and ':' in kek_map  # Atom map numbers present

def test_get_hybridization_indices_basic():
    from rdkit import Chem
    from src.llm.llm_mol import get_hybridization_indices
    mol = Chem.MolFromSmiles('CC=C')
    hyb = get_hybridization_indices(mol)
    assert isinstance(hyb, list)
    assert len(hyb) == 3
    # sp3, sp2, sp
    assert any(isinstance(x, list) for x in hyb)

def test_get_element_atom_indices_basic():
    from rdkit import Chem
    from src.llm.llm_mol import get_element_atom_indices
    mol = Chem.MolFromSmiles('FCNOPSCl')
    indices = get_element_atom_indices(mol)
    assert len(indices) == 7
    # Each element should have one atom index
    for lst in indices:
        assert len(lst) == 1

def test_get_bond_counts_basic():
    from rdkit import Chem
    from src.llm.llm_mol import get_bond_counts
    mol = Chem.MolFromSmiles('CC#N')
    counts = get_bond_counts(mol)
    assert isinstance(counts, list)
    assert len(counts) == 3
    assert counts[1] == 0  # No double bonds
    assert counts[2] == 1  # One triple bond

def test_get_ring_counts_basic():
    from rdkit import Chem
    from src.llm.llm_mol import get_ring_counts
    mol = Chem.MolFromSmiles('c1ccccc1')
    counts = get_ring_counts(mol)
    assert isinstance(counts, list)
    assert len(counts) == 4
    assert counts[0] == 1  # One ring
    assert counts[1] == 1  # One aromatic ring

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


# =================================================
# Tests for _norm_tagged function
# =================================================

def test_norm_tagged_basic():
    """Test basic functionality of _norm_tagged with simple answer tags."""
    # Basic answer extraction
    assert _norm_tagged("<answer>42</answer>") == "42"
    assert _norm_tagged("<answer>ethanol</answer>") == "ethanol"
    assert _norm_tagged("<answer>benzene</answer>") == "benzene"
    
    # With surrounding text
    text = "The molecular formula is <answer>C6H6</answer> for benzene."
    assert _norm_tagged(text) == "C6H6"

def test_norm_tagged_with_periods():
    """Test that trailing periods are removed from tagged answers."""
    # Period removal
    assert _norm_tagged("<answer>42.</answer>") == "42"
    assert _norm_tagged("<answer>ethanol.</answer>") == "ethanol"
    assert _norm_tagged("<answer>C2H6O.</answer>") == "C2H6O"
    
    # Multiple periods
    assert _norm_tagged("<answer>answer...</answer>") == "answer.."
    
    # Period in middle should be preserved
    assert _norm_tagged("<answer>3.14</answer>") == "3.14"
    assert _norm_tagged("<answer>3.14.</answer>") == "3.14"

def test_norm_tagged_multiline():
    """Test _norm_tagged with multiline content between tags."""
    multiline_text = """
    The answer is:
    <answer>
    6
    </answer>
    This is the final result.
    """
    assert _norm_tagged(multiline_text) == "6"
    
    # Multiline with more complex content
    complex_multiline = """<answer>
    The molecule has:
    - 6 carbon atoms
    - 6 hydrogen atoms
    Final answer: benzene
    </answer>"""
    expected = "The molecule has:\n    - 6 carbon atoms\n    - 6 hydrogen atoms\n    Final answer: benzene"
    assert _norm_tagged(complex_multiline) == expected

def test_norm_tagged_whitespace_handling():
    """Test proper whitespace handling in tagged answers."""
    # Leading/trailing whitespace inside tags
    assert _norm_tagged("<answer>  42  </answer>") == "42"
    assert _norm_tagged("<answer>\n  ethanol  \n</answer>") == "ethanol"
    assert _norm_tagged("<answer>\t\tbenzene\t\t</answer>") == "benzene"
    
    # Whitespace with periods
    assert _norm_tagged("<answer>  42.  </answer>") == "42"
    assert _norm_tagged("<answer>\n  ethanol.  \n</answer>") == "ethanol"

def test_norm_tagged_no_tags():
    """Test fallback behavior when no answer tags are found."""
    # No tags - should return stripped original string
    assert _norm_tagged("42") == "42"
    assert _norm_tagged("  ethanol  ") == "ethanol"
    assert _norm_tagged("No answer tags here") == "No answer tags here"
    assert _norm_tagged("") == ""
    assert _norm_tagged("   ") == ""

def test_norm_tagged_malformed_tags():
    """Test behavior with malformed or incomplete tags."""
    # Missing closing tag
    assert _norm_tagged("<answer>42") == "<answer>42"
    
    # Missing opening tag
    assert _norm_tagged("42</answer>") == "42</answer>"
    
    # Empty tags
    assert _norm_tagged("<answer></answer>") == ""
    
    # Tags with only whitespace
    assert _norm_tagged("<answer>   </answer>") == ""
    assert _norm_tagged("<answer>\n\t  \n</answer>") == ""

def test_norm_tagged_multiple_tags():
    """Test behavior when multiple answer tags are present."""
    # Multiple tags - should extract from first match
    text = "First <answer>42</answer> and second <answer>24</answer>"
    assert _norm_tagged(text) == "24"
    
def test_norm_tagged_special_content():
    """Test _norm_tagged with special characters and content types."""
    # Numbers
    assert _norm_tagged("<answer>3.14159</answer>") == "3.14159"
    assert _norm_tagged("<answer>-42</answer>") == "-42"
    assert _norm_tagged("<answer>1.5e-10</answer>") == "1.5e-10"
    
    # Chemical formulas
    assert _norm_tagged("<answer>C6H12O6</answer>") == "C6H12O6"
    assert _norm_tagged("<answer>Ca(OH)2</answer>") == "Ca(OH)2"
    
    # SMILES strings
    assert _norm_tagged("<answer>CCO</answer>") == "CCO"
    assert _norm_tagged("<answer>c1ccccc1</answer>") == "c1ccccc1"
    
    # IUPAC names
    assert _norm_tagged("<answer>2-methylbutane</answer>") == "2-methylbutane"
    assert _norm_tagged("<answer>N,N-dimethylmethanamine</answer>") == "N,N-dimethylmethanamine"

def test_norm_tagged_case_sensitivity():
    """Test that _norm_tagged is case sensitive for tags."""
    # Correct case
    assert _norm_tagged("<answer>42</answer>") == "42"
    
    # Wrong case - should not match
    assert _norm_tagged("<ANSWER>42</ANSWER>") == "<ANSWER>42</ANSWER>"
    assert _norm_tagged("<Answer>42</Answer>") == "<Answer>42</Answer>"

def test_norm_tagged_tokenizer_parameter():
    """Test that tokenizer parameter is handled correctly (though unused)."""
    # Should work the same regardless of tokenizer parameter
    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token = "<|im_end|>"
    
    assert _norm_tagged("<answer>42</answer>") == "42"
    assert _norm_tagged("<answer>42</answer>", tokenizer=None) == "42"
    assert _norm_tagged("<answer>42</answer>", tokenizer=mock_tokenizer) == "42"
    
    # With periods
    assert _norm_tagged("<answer>42.</answer>", tokenizer=mock_tokenizer) == "42"

def test_norm_tagged_realistic_llm_outputs():
    """Test with realistic LLM outputs that might contain answer tags."""
    # Typical LLM response with explanation
    response1 = """
    To find the number of carbon atoms, I need to analyze the molecular structure.
    
    Looking at the SMILES string CCO:
    - First C is a carbon atom
    - Second C is a carbon atom  
    - O is an oxygen atom
    
    <answer>2</answer>
    
    Therefore, there are 2 carbon atoms in this molecule.
    """
    assert _norm_tagged(response1) == "2"
    
    # Response with reasoning and period
    response2 = """
    The IUPAC name for this compound can be determined by:
    1. Identifying the longest carbon chain
    2. Numbering the chain
    3. Naming substituents
    
    <answer>ethanol.</answer>
    """
    assert _norm_tagged(response2) == "ethanol"
    
    # Response with complex chemical answer
    response3 = """
    The molecular formula is derived from the structure:
    <answer>C6H12O6</answer>
    This represents glucose.
    """
    assert _norm_tagged(response3) == "C6H12O6"


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
    assert props["positive_formal_charge_count"] == [0, 1]
    assert props["negative_formal_charge_count"] == [0, 0]
    # Check new stereo_summary output
    assert props["stereo_summary"][0][0] == 0  # CCO: no stereocenter
    assert props["stereo_summary"][0][1] == 0  # CCO: no stereo bond
    assert props["stereo_summary"][1][0] == 0  # [NH4+]: no stereocenter
    assert props["stereo_summary"][1][1] == 0  # [NH4+]: no stereo bond

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
    assert answers[0]["iupac_name"] == ["ethanol", "acetic acid"]
    # Check new stereo_summary output
    assert answers[0]["stereo_summary"][0] == '[0, 0]'  # ethanol: no stereocenter
    assert answers[0]["stereo_summary"][1] == '[0, 0]'  # acetic acid: no stereocenter

def test_all_properties_processor():
    smiles = ["CCO", "CC(=O)O"]
    iupac = ["ethanol", "acetic acid"]
    table = pa.table({"smiles": smiles, "iupac": iupac})
    proc = AllPropertiesProcessor()
    answers = proc.prepare_answers(table)
    assert answers[0]["iupac_name"] == ["ethanol", "acetic acid"]
    # Check other properties exist
    assert "heavy_atom_count" in answers[0]
    # Check new stereo_summary output
    assert answers[0]["stereo_summary"][0] == '[0, 0]'  # ethanol: no stereocenter
    assert answers[0]["stereo_summary"][1] == '[0, 0]'  # acetic acid: no stereocenter
