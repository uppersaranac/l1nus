import pytest
import numpy as np
import torch
from unittest.mock import MagicMock
from transformers import AutoTokenizer
from llm.llm_apis import (
    build_train_batch, build_eval_batch, _norm, compute_metrics_closure, do_generation,
    count_heavy_atoms, count_non_hydrogen_bonds, count_positive_formal_charge_atoms, count_negative_formal_charge_atoms,
    calculate_molecular_properties,
    QuestionSetProcessor, IUPACNamingProcessor, MolecularPropertiesProcessor, AllPropertiesProcessor,
    QUESTION_SETS
)

@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token
    return tok

@pytest.fixture
def toy_data():
    return ["CCO", "CC(=O)O"], ["ethanol", "acetic acid"]

def test_build_train_batch(tokenizer, toy_data):
    smiles, iupac = toy_data
    max_len = 32
    processor = IUPACNamingProcessor()
    answers = processor.prepare_answers({"smiles": smiles, "iupac": iupac})
    batch = build_train_batch(tokenizer, smiles, answers, max_len=max_len)

    input_ids = batch["input_ids"]
    labels = batch["labels"]

    assert len(input_ids) == len(smiles)
    assert all(len(seq) == max_len for seq in input_ids)
    assert all(len(seq) == max_len for seq in labels)

    for inp, lab in zip(input_ids, labels):
        padding_len = sum(1 for tok in inp if tok == tokenizer.pad_token_id)
        assert all(tok == -100 for tok in lab[:padding_len])
        for i in range(max_len):
            if lab[i] != -100:
                assert lab[i] == inp[i]

def test_build_eval_batch(tokenizer, toy_data):
    smiles, iupac = toy_data
    max_prompt_len = 32
    max_label_len = 16
    processor = IUPACNamingProcessor()
    answers = processor.prepare_answers({"smiles": smiles, "iupac": iupac})
    batch = build_eval_batch(tokenizer, smiles, answers, max_prompt_len, max_label_len)

    input_ids = batch["input_ids"]
    labels = batch["labels"]

    assert len(input_ids) == len(smiles)
    assert all(len(seq) == max_prompt_len for seq in input_ids)
    assert all(len(seq) == max_prompt_len for seq in labels)

    for label in labels:
        assert all(tok == -100 for tok in label[:max_prompt_len - max_label_len])
        assert any(tok != -100 for tok in label[max_prompt_len - max_label_len:])

def test_norm():
    assert _norm("It is <result>ethanol</result>") == "ethanol"
    assert _norm(" <result>ethanol</result> ") == "ethanol"
    assert _norm("It is \\boxed{3}") == "3"

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
    ds = {"smiles": ["CCO", "CC(=O)O"], "iupac": ["ethanol", "acetic acid"]}
    proc = IUPACNamingProcessor()
    answers = proc.prepare_answers(ds)
    assert answers == {"iupac_name": ["ethanol", "acetic acid"]}
    q = QUESTION_SETS["iupac_naming"]["questions"][0]
    for i, smile in enumerate(ds["smiles"]):
        formatted = proc.format_answer(q, answers, i, smile)
        assert ds["iupac"][i] in formatted


def test_molecular_properties_processor():
    ds = {"smiles": ["CCO", "CC(=O)O"], "iupac": ["ethanol", "acetic acid"]}
    proc = MolecularPropertiesProcessor()
    answers = proc.prepare_answers(ds)
    # Check a few key properties for both molecules
    assert answers["carbon_count"] == [2, 2]
    assert answers["oxygen_count"] == [1, 2]
    assert answers["iupac_name"] == ["ethanol", "acetic acid"]
    q = QUESTION_SETS["molecular_properties"]["questions"][0]
    for i, smile in enumerate(ds["smiles"]):
        formatted = proc.format_answer(q, answers, i, smile)
        # Should contain the correct answer for each molecule
        assert str(answers[q["id"]][i]) in formatted


def test_all_properties_processor():
    ds = {"smiles": ["CCO", "CC(=O)O"], "iupac": ["ethanol", "acetic acid"]}
    proc = AllPropertiesProcessor()
    answers = proc.prepare_answers(ds)
    q = QUESTION_SETS["all_properties"]["questions"][0]
    for i, smile in enumerate(ds["smiles"]):
        formatted = proc.format_answer(q, answers, i, smile)
        # Should include the assistant template's header and at least one property value
        assert "Molecular Analysis" in formatted
        assert str(answers["carbon_count"][i]) in formatted
        assert ds["iupac"][i] in formatted


def test_processors_edge_cases():
    # Edge case 1: Empty input
    ds_empty = {"smiles": [], "iupac": []}
    for Processor in [IUPACNamingProcessor, MolecularPropertiesProcessor, AllPropertiesProcessor]:
        proc = Processor()
        answers = proc.prepare_answers(ds_empty)
        # All returned lists should be empty
        for v in answers.values():
            assert v == []

    # Edge case 2: Invalid SMILES
    ds_invalid = {"smiles": ["", "not_a_smiles"], "iupac": ["", "invalid"]}
    # Should not raise, and should return zero/empty/placeholder values
    for Processor in [MolecularPropertiesProcessor, AllPropertiesProcessor]:
        proc = Processor()
        answers = proc.prepare_answers(ds_invalid)
        # All property lists should have correct length and handle invalid gracefully
        for v in answers.values():
            assert len(v) == 2
    # IUPACNamingProcessor just returns the iupac field
    proc = IUPACNamingProcessor()
    answers = proc.prepare_answers(ds_invalid)
    assert answers["iupac_name"] == ["", "invalid"]

    # Edge case 3: Mismatched lengths
    ds_mismatch = {"smiles": ["CCO"], "iupac": ["ethanol", "extra"]}
    for Processor in [IUPACNamingProcessor, MolecularPropertiesProcessor, AllPropertiesProcessor]:
        proc = Processor()
        try:
            answers = proc.prepare_answers(ds_mismatch)
        except Exception as e:
            # Should raise a clear error or handle gracefully
            assert isinstance(e, (IndexError, ValueError, AssertionError))

def test_molecular_properties_processor():
    ds = {"smiles": ["CCO"], "iupac": ["ethanol"]}
    proc = MolecularPropertiesProcessor()
    answers = proc.prepare_answers(ds)
    assert answers["carbon_count"][0] == 2
    q = QUESTION_SETS["molecular_properties"]["questions"][0]
    formatted = proc.format_answer(q, answers, 0, "CCO")
    assert str(answers[q["id"]][0]) in formatted

def test_all_properties_processor():
    ds = {"smiles": ["CCO"], "iupac": ["ethanol"]}
    proc = AllPropertiesProcessor()
    answers = proc.prepare_answers(ds)
    q = QUESTION_SETS["all_properties"]["questions"][0]
    formatted = proc.format_answer(q, answers, 0, "CCO")
    assert "Molecular Analysis" in formatted
    assert str(answers["carbon_count"][0]) in formatted

def test_question_set_processor_not_implemented():
    class Dummy(QuestionSetProcessor):
        pass
    dummy = Dummy("iupac_naming")
    with pytest.raises(NotImplementedError):
        dummy.prepare_answers({})
    with pytest.raises(NotImplementedError):
        dummy.format_answer({}, {}, 0, "CCO")

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

    preds = do_generation(num_beams=1, max_new_tokens=5,
                          tokenizer=tokenizer, model=model,
                          data=DummyDataset())
    import numpy as np
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == 2
    assert preds.ndim == 2
    assert preds.dtype == np.int64
    model.generate.assert_called_once()