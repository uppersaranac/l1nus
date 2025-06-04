#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path
import random
from typing import Any, Sequence, Dict, Optional, Callable

import numpy as np
import evaluate
from datasets import Dataset
import torch
from transformers import TrainerCallback

# Import rdkit for molecular property calculations
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

def count_heavy_atoms(smiles: str) -> int:
    """
    Count the number of heavy (non-hydrogen) atoms in a molecule.

    :param smiles: SMILES string representation of the molecule.
    :return: Number of heavy atoms in the molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return mol.GetNumHeavyAtoms()

def count_non_hydrogen_bonds(smiles: str) -> int:
    """
    Count the number of bonds not to hydrogen in a molecule.

    :param smiles: SMILES string representation of the molecule.
    :return: Number of bonds not to hydrogen in the molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return sum(1 for bond in mol.GetBonds() if bond.GetBeginAtom().GetSymbol() != 'H' and bond.GetEndAtom().GetSymbol() != 'H')

def count_positive_formal_charge_atoms(smiles: str) -> int:
    """
    Count the number of atoms with positive formal charge in a molecule.

    :param smiles: SMILES string representation of the molecule.
    :return: Number of atoms with positive formal charge.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetFormalCharge() > 0)

def count_negative_formal_charge_atoms(smiles: str) -> int:
    """
    Count the number of atoms with negative formal charge in a molecule.

    :param smiles: SMILES string representation of the molecule.
    :return: Number of atoms with negative formal charge.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetFormalCharge() < 0)



SYSTEM_PROMPT = "Do not think."

# Processor to handle different question sets
class QuestionSetProcessor:
    """
    Base class for handling answer preparation and example display for a question set.

    :param name: Name of the question set.
    """
    def __init__(self, name: str):
        self.name = name
        self.questions = QUESTION_SETS[name]["questions"]

    def prepare_answers(self, ds: Any) -> Dict[str, Sequence[Any]]:
        """
        Prepare answers for the question set from a dataset.

        :param ds: Dataset containing SMILES and possibly IUPAC names.
        :return: Dictionary mapping property/question names to lists of answers.
        """
        raise NotImplementedError

    def show_examples(self, ds: Any, tok: Any, n: int) -> None:
        """
        Print example questions and answers for the question set.

        :param ds: Dataset containing SMILES and answers.
        :param tok: Tokenizer (currently unused).
        :param n: Number of examples to show.
        """
        answers = self.prepare_answers(ds)
        for i in range(min(len(ds), n)):
            smile = ds["smiles"][i]
            for q in self.questions:
                q_str = q["user_template"].format(smiles=smile)
                a_str = self.format_answer(q, answers, i, smile)
                print(f"Q: {q_str}")
                print(f"A: {a_str}")
            print("-" * 40)

    def format_answer(self, q: dict, answers: dict, i: int, smile: str) -> str:
        """
        Format the answer string for a specific question and molecule.

        :param q: Question dictionary (with templates).
        :param answers: Dictionary of answers for all molecules.
        :param i: Index of the molecule in the dataset.
        :param smile: SMILES string for the molecule.
        :return: Formatted answer string.
        """
        raise NotImplementedError
        
    def expand_dataset(self, ds: Any, answers: Dict[str, Sequence[Any]]):
        """
        Expand a dataset to include all question/answer pairs.
        
        :param ds: Dataset containing SMILES and answers.
        :param answers: Dictionary of prepared answers.
        :return: Expanded dataset with one entry per Q&A pair.
        """
        from datasets import Dataset
        
        expanded_data = {
            "smiles": [],
            "question_id": [],
            "question_template": [],
            "answer": [],
            "assistant_template": [],
            "system_prompt": []
        }
        
        system_prompt = QUESTION_SETS[self.name]["system_prompt"]
        
        for i, smile in enumerate(ds["smiles"]):
            for q in self.questions:
                q_id = q["id"]
                # Check if we have an answer for this question and molecule
                if q_id in answers and i < len(answers[q_id]):
                    expanded_data["smiles"].append(smile)
                    expanded_data["question_id"].append(q_id)
                    expanded_data["question_template"].append(q["user_template"])
                    expanded_data["answer"].append(answers[q_id][i])
                    expanded_data["assistant_template"].append(q["assistant_template"])
                    expanded_data["system_prompt"].append(system_prompt)
        
        return Dataset.from_dict(expanded_data)

class IUPACNamingProcessor(QuestionSetProcessor):
    """
    Processor for the IUPAC naming question set.
    """
    def __init__(self):
        super().__init__("iupac_naming")
    def prepare_answers(self, ds: Any) -> Dict[str, Sequence[Any]]:
        """
        Prepare answers for IUPAC naming (just returns the IUPAC names).

        :param ds: Dataset containing IUPAC names.
        :return: Dictionary with 'iupac_name' mapped to the list of names.
        """
        return {"iupac_name": ds["iupac"]}
    def format_answer(self, q: dict, answers: dict, i: int, smile: str) -> str:
        ans = answers[q["id"]][i]
        return q["assistant_template"].format(answer=ans)

class MolecularPropertiesProcessor(QuestionSetProcessor):
    """
    Processor for the molecular properties question set.
    """
    def __init__(self):
        super().__init__("molecular_properties")
    def prepare_answers(self, ds: Any) -> Dict[str, Sequence[Any]]:
        answers = calculate_molecular_properties(ds["smiles"])
        answers["iupac_name"] = ds["iupac"]
        # Ensure all answers are lists of strings
        for k in answers:
            answers[k] = [str(x) for x in answers[k]]
        return answers
    def format_answer(self, q: dict, answers: dict, i: int, smile: str) -> str:
        ans = answers[q["id"]][i]
        return q["assistant_template"].format(answer=ans)

class AllPropertiesProcessor(QuestionSetProcessor):
    """
    Processor for the comprehensive 'all_properties' question set.
    """
    def __init__(self):
        super().__init__("all_properties")
    def prepare_answers(self, ds: Any) -> Dict[str, Sequence[Any]]:
        answers = calculate_molecular_properties(ds["smiles"])
        answers["iupac_name"] = ds["iupac"]
        # Ensure all answers are lists of strings
        for k in answers:
            answers[k] = [str(x) for x in answers[k]]
        return answers
    def format_answer(self, q: dict, answers: dict, i: int, smile: str) -> str:
        """
        Format the answer for the all-properties question (expands all properties into the template).

        :param q: Question dictionary.
        :param answers: Dictionary of answers.
        :param i: Index of the molecule.
        :param smile: SMILES string.
        :return: Formatted answer string.
        """
        props = {key: answers[key][i] for key in answers}
        return q["assistant_template"].format(smiles=smile, **props)

# Mapping of question set names to their processor classes.
#: Dict[str, Type[QuestionSetProcessor]]
PROCESSOR_CLASSES = {
    "iupac_naming": IUPACNamingProcessor,
    "molecular_properties": MolecularPropertiesProcessor,
    "all_properties": AllPropertiesProcessor,
}

# =================================================
# Molecular Property Functions
# =================================================

def count_element_atoms(mol: Any, element: str) -> int:
    """
    Count the number of atoms of a specific element in a molecule.

    :param mol: RDKit molecule object.
    :param element: Chemical symbol of the element to count.
    :return: Number of atoms of the given element in the molecule.
    """
    return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == element)

def count_carbon_atoms(smiles: str) -> int:
    """
    Count carbon atoms in a molecule.

    :param smiles: SMILES string representation of the molecule.
    :return: Number of carbon atoms in the molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return count_element_atoms(mol, 'C')

def count_nitrogen_atoms(smiles: str) -> int:
    """
    Count nitrogen atoms in a molecule.

    :param smiles: SMILES string representation of the molecule.
    :return: Number of nitrogen atoms in the molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return count_element_atoms(mol, 'N')

def count_oxygen_atoms(smiles: str) -> int:
    """
    Count oxygen atoms in a molecule.

    :param smiles: SMILES string representation of the molecule.
    :return: Number of oxygen atoms in the molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return count_element_atoms(mol, 'O')

def count_sulfur_atoms(smiles: str) -> int:
    """
    Count sulfur atoms in a molecule.

    :param smiles: SMILES string representation of the molecule.
    :return: Number of sulfur atoms in the molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return count_element_atoms(mol, 'S')

def count_phosphorus_atoms(smiles: str) -> int:
    """
    Count phosphorus atoms in a molecule.

    :param smiles: SMILES string representation of the molecule.
    :return: Number of phosphorus atoms in the molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return count_element_atoms(mol, 'P')

def count_chlorine_atoms(smiles: str) -> int:
    """
    Count chlorine atoms in a molecule.

    :param smiles: SMILES string representation of the molecule.
    :return: Number of chlorine atoms in the molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return count_element_atoms(mol, 'Cl')

def count_fluorine_atoms(smiles: str) -> int:
    """
    Count fluorine atoms in a molecule.

    :param smiles: SMILES string representation of the molecule.
    :return: Number of fluorine atoms in the molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return count_element_atoms(mol, 'F')

def count_rings(smiles: str) -> int:
    """
    Count the number of rings in a molecule.

    :param smiles: SMILES string representation of the molecule.
    :return: Number of rings in the molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return rdMolDescriptors.CalcNumRings(mol)

def count_aromatic_rings(smiles: str) -> int:
    """
    Count the number of aromatic rings in a molecule given its SMILES string.

    :param smiles: SMILES string of the molecule.
    :return: Number of aromatic rings.
    """
    """
    Count the number of aromatic rings in a molecule.

    :param smiles: SMILES string representation of the molecule.
    :return: Number of aromatic rings in the molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    count = 0
    for ring in Chem.GetSymmSSSR(mol):
        atoms = list(ring)
        if all(
            mol.GetBondBetweenAtoms(a1, a2).GetIsAromatic()
            for a1, a2 in zip(atoms, atoms[1:] + [atoms[0]])
        ):
            count += 1
    return count

def count_double_bonds(smiles: str) -> int:
    """
    Count the number of double bonds in a molecule.

    :param smiles: SMILES string representation of the molecule.
    :return: Number of double bonds in the molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE)

def count_triple_bonds(smiles: str) -> int:
    """
    Count the number of triple bonds in a molecule.

    :param smiles: SMILES string representation of the molecule.
    :return: Number of triple bonds in the molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.TRIPLE)

def count_stereo_double_bonds(smiles: str) -> int:
    """
    Count the number of stereo double bonds (E/Z) in a molecule.

    :param smiles: SMILES string representation of the molecule.
    :return: Number of stereo (E/Z) double bonds in the molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return sum(1 for bond in mol.GetBonds() 
               if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE 
               and (bond.GetStereo() == Chem.rdchem.BondStereo.STEREOE or 
                    bond.GetStereo() == Chem.rdchem.BondStereo.STEREOZ))

def count_stereocenters(smiles: str) -> int:
    """
    Count the number of stereocenters in a molecule.

    :param smiles: SMILES string representation of the molecule.
    :return: Number of stereocenters in the molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return rdMolDescriptors.CalcNumAtomStereoCenters(mol)

def count_positive_formal_charge_atoms(smiles: str) -> int:
    """
    Count the number of atoms with positive formal charge in a molecule.

    :param smiles: SMILES string representation of the molecule.
    :return: Number of atoms with positive formal charge in the molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetFormalCharge() > 0)

def count_negative_formal_charge_atoms(smiles: str) -> int:
    """
    Count the number of atoms with negative formal charge in a molecule.

    :param smiles: SMILES string representation of the molecule.
    :return: Number of atoms with negative formal charge in the molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetFormalCharge() < 0)

# Function to calculate all properties for a set of molecules
def calculate_molecular_properties(smiles_list: Sequence[str]) -> Dict[str, Sequence[Any]]:
    """
    Calculate various molecular properties for a list of SMILES strings.

    :param smiles_list: List of SMILES strings.
    :return: Dictionary mapping property names to lists of property values.
    """
    return {
        "carbon_count": [count_carbon_atoms(s) for s in smiles_list],
        "nitrogen_count": [count_nitrogen_atoms(s) for s in smiles_list],
        "oxygen_count": [count_oxygen_atoms(s) for s in smiles_list],
        "sulfur_count": [count_sulfur_atoms(s) for s in smiles_list],
        "phosphorus_count": [count_phosphorus_atoms(s) for s in smiles_list],
        "chlorine_count": [count_chlorine_atoms(s) for s in smiles_list],
        "fluorine_count": [count_fluorine_atoms(s) for s in smiles_list],
        "ring_count": [count_rings(s) for s in smiles_list],
        "aromatic_ring_count": [count_aromatic_rings(s) for s in smiles_list],
        "double_bond_count": [count_double_bonds(s) for s in smiles_list],
        "triple_bond_count": [count_triple_bonds(s) for s in smiles_list],
        "stereo_double_bond_count": [count_stereo_double_bonds(s) for s in smiles_list],
        "stereocenter_count": [count_stereocenters(s) for s in smiles_list],
        "heavy_atom_count": [count_heavy_atoms(s) for s in smiles_list],
        "non_hydrogen_bond_count": [count_non_hydrogen_bonds(s) for s in smiles_list],
        "positive_formal_charge_count": [count_positive_formal_charge_atoms(s) for s in smiles_list],
        "negative_formal_charge_count": [count_negative_formal_charge_atoms(s) for s in smiles_list],
    }

# =================================================
# Question Sets and Templates
# =================================================

# Define different question sets
QUESTION_SETS = {
    "iupac_naming": {
        "system_prompt": SYSTEM_PROMPT+" Place the answer between <|extra_100|> and <|extra_101|>.",
        "questions": [
            {
                "id": "iupac_name",
                "user_template": "Use the IUPAC naming rules to name the molecule {smiles}.",
                "assistant_template": "<|extra_100|>{answer}<|extra_101|>."
            }
        ]
    },
    "molecular_properties": {
        "system_prompt": SYSTEM_PROMPT+" Place the answer between <|extra_100|> and <|extra_101|>.  The answer should be a number.",
        "questions": [
            {
                "id": "carbon_count",
                "user_template": "How many carbon atoms are in the molecule {smiles}?",
                "assistant_template": "<|extra_100|>{answer}<|extra_101|>"
            },
            {
                "id": "heavy_atom_count",
                "user_template": "How many heavy (non-hydrogen) atoms are in the molecule {smiles}?",
                "assistant_template": "<|extra_100|>{answer}<|extra_101|>"
            },
            {
                "id": "non_hydrogen_bond_count",
                "user_template": "How many bonds not to hydrogen are in the molecule {smiles}?",
                "assistant_template": "<|extra_100|>{answer}<|extra_101|>"
            },
            {
                "id": "positive_formal_charge_count",
                "user_template": "How many atoms with positive formal charge are in the molecule {smiles}?",
                "assistant_template": "<|extra_100|>{answer}<|extra_101|>"
            },
            {
                "id": "negative_formal_charge_count",
                "user_template": "How many atoms with negative formal charge are in the molecule {smiles}?",
                "assistant_template": "<|extra_100|>{answer}<|extra_101|>"
            },
            {
                "id": "nitrogen_count",
                "user_template": "How many nitrogen atoms are in the molecule {smiles}?",
                "assistant_template": "<|extra_100|>{answer}<|extra_101|>"
            },
            {
                "id": "oxygen_count",
                "user_template": "How many oxygen atoms are in the molecule {smiles}?",
                "assistant_template": "<|extra_100|>{answer}<|extra_101|>"
            },
            {
                "id": "sulfur_count",
                "user_template": "How many sulfur atoms are in the molecule {smiles}?",
                "assistant_template": "<|extra_100|>{answer}<|extra_101|>"
            },
            {
                "id": "phosphorus_count",
                "user_template": "How many phosphorus atoms are in the molecule {smiles}?", 
                "assistant_template": "<|extra_100|>{answer}<|extra_101|>"
            },
            {
                "id": "chlorine_count", 
                "user_template": "How many chlorine atoms are in the molecule {smiles}?",
                "assistant_template": "<|extra_100|>{answer}<|extra_101|>"
            },
            {
                "id": "fluorine_count",
                "user_template": "How many fluorine atoms are in the molecule {smiles}?",
                "assistant_template": "<|extra_100|>{answer}<|extra_101|>"
            },
            {
                "id": "ring_count",
                "user_template": "How many rings are in the molecule {smiles}?",
                "assistant_template": "<|extra_100|>{answer}<|extra_101|>"
            },
            {
                "id": "aromatic_ring_count",
                "user_template": "How many aromatic rings are in the molecule {smiles}?",
                "assistant_template": "<|extra_100|>{answer}<|extra_101|>"
            },
            {
                "id": "double_bond_count",
                "user_template": "How many double bonds are in the molecule {smiles}?",
                "assistant_template": "<|extra_100|>{answer}<|extra_101|>"
            },
            {
                "id": "triple_bond_count",
                "user_template": "How many triple bonds are in the molecule {smiles}?",
                "assistant_template": "<|extra_100|>{answer}<|extra_101|>"
            },
            {
                "id": "stereo_double_bond_count",
                "user_template": "How many stereo double bonds are in the molecule {smiles}?",
                "assistant_template": "<|extra_100|>{answer}<|extra_101|>"
            },
            {
                "id": "stereocenter_count",
                "user_template": "How many stereocenters are in the molecule {smiles}?",
                "assistant_template": "<|extra_100|>{answer}<|extra_101|>"
            }
        ]
    },
    "all_properties": {
        "system_prompt": SYSTEM_PROMPT+" Put the answers in <|extra_100|> and <|extra_101|>.",
        "questions": [
            {
                "id": "all_properties",
                "user_template": "Analyze the following molecular properties for the molecule {smiles}: carbon atoms, nitrogen atoms, oxygen atoms, sulfur atoms, phosphorus atoms, chlorine atoms, fluorine atoms, rings, aromatic rings, double bonds, triple bonds, stereo double bonds, stereocenters, and the IUPAC name. Provide a comprehensive report.",
                "assistant_template": "Molecular Analysis of {smiles}:\n\nCarbon atoms: <|extra_100|>{carbon_count}<|extra_101|>\nNitrogen atoms: <|extra_100|>{nitrogen_count}<|extra_101|>\nOxygen atoms: <|extra_100|>{oxygen_count}<|extra_101|>\nSulfur atoms: <|extra_100|>{sulfur_count}<|extra_101|>\nPhosphorus atoms: <|extra_100|>{phosphorus_count}<|extra_101|>\nChlorine atoms: <|extra_100|>{chlorine_count}<|extra_101|>\nFluorine atoms: <|extra_100|>{fluorine_count}<|extra_101|>\nRings: <|extra_100|>{ring_count}<|extra_101|>\nAromatic rings: <|extra_100|>{aromatic_ring_count}<|extra_101|>\nDouble bonds: <|extra_100|>{double_bond_count}<|extra_101|>\nTriple bonds: <|extra_100|>{triple_bond_count}<|extra_101|>\nStereo double bonds: <|extra_100|>{stereo_double_bond_count}<|extra_101|>\nStereocenters: <|extra_100|>{stereocenter_count}<|extra_101|>\nHeavy atoms: <|extra_100|>{heavy_atom_count}<|extra_101|>\nBonds not to hydrogen: <|extra_100|>{non_hydrogen_bond_count}<|extra_101|>\nAtoms with positive formal charge: <|extra_100|>{positive_formal_charge_count}<|extra_101|>\nAtoms with negative formal charge: <|extra_100|>{negative_formal_charge_count}<|extra_101|>\nIUPAC name: <|extra_100|>{iupac_name}<|extra_101|>"
            }
        ]
    }
}

# ─────────────────────────── data helper ──────────────────────────────
def load_arrow_dataset(path: str, limit: Optional[int] = None) -> Dataset:
    """
    Load an Arrow dataset from a file.

    :param path: Path to the dataset file.
    :param limit: Optional limit on the number of examples to load.
    :return: Loaded dataset.
    """
    ds = Dataset.from_file(str(Path(path).expanduser()))
    if limit and limit > 0 and len(ds) > limit:
        ds = ds.select(random.sample(range(len(ds)), limit))
        # ds = ds.select(range(limit))
    return ds


def process_single_qa(tok: Any, example: Dict[str, Any], max_len: int, max_label_len: int = None, is_train: bool = True) -> Dict[str, Any]:
    """
    Process a single question-answer pair from the expanded dataset.
    
    :param tok: Tokenizer instance
    :param example: Dictionary containing a single Q&A pair with all necessary fields
    :param max_len: Maximum length for the input
    :param max_label_len: Maximum length for the label (only used for eval)
    :param is_train: Whether this is for training or evaluation
    :return: Dictionary with tokenized input_ids, attention_mask, and labels
    """
    # Build the prompt
    if is_train:
        # For training, we include both question and answer
        if hasattr(tok, 'apply_chat_template'):
            # Use chat template if available
            prompt = [
                {"role": "system", "content": example["system_prompt"]},
                {"role": "user", "content": example["question_template"].format(smiles=example["smiles"])},
                {"role": "assistant", "content": example["assistant_template"].format(answer=example["answer"])}
            ]
            prompt_str = tok.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
        else:
            # Fallback for models without chat templates
            system = example["system_prompt"]
            question = example["question_template"].format(smiles=example["smiles"])
            answer = example["assistant_template"].format(answer=example["answer"])
            prompt_str = f"{system}\n\nuser: {question}\n\nassistant: {answer}"
    else:
        # For evaluation, only include the question (no answer)
        if hasattr(tok, 'apply_chat_template'):
            prompt = [  
                {"role": "system", "content": example["system_prompt"]},
                {"role": "user", "content": example["question_template"].format(smiles=example["smiles"])}
            ]
            prompt_str = tok.apply_chat_template(prompt, add_generation_prompt=True, 
                                                tokenize=False, enable_thinking=False)
        else:
            system = example["system_prompt"]
            question = example["question_template"].format(smiles=example["smiles"])
            prompt_str = f"{system}\n\nuser: {question}\n\nassistant: "
    
    # Tokenize the prompt
    tokenized_output = tok(prompt_str, padding="max_length", truncation=True, max_length=max_len, return_tensors="np")
    
    # Ensure input_ids and attention_mask are 1D
    # The tokenizer returns (1, seq_len) for single examples, so we take the first element.
    input_ids = tokenized_output["input_ids"][0]
    attention_mask = tokenized_output["attention_mask"][0]
    
    processed_example = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    
    if is_train:
        # For training: find the answer span in the prompt
        answer_text = str(example["assistant_template"].format(answer=example["answer"]))+"<|im_end|>"
        
        input_ids_list = input_ids.tolist() # Already 1D
        
        # Use robust helper to find answer token positions
        answer_span = find_answer_token_positions(tok, prompt_str, answer_text, input_ids_list, max_len)
        label = [-100] * len(input_ids_list)
        if answer_span is not None:
            start_idx, end_idx = answer_span
            for j in range(start_idx, end_idx):
                label[j] = input_ids_list[j]
        else:
            print(f"Warning: Answer not found in input_ids for example with answer {answer_text}")
        processed_example["labels"] = label  # Assign 1D list directly
    else:
        # For evaluation: right-align answer tokens
        formatted_answer = example["assistant_template"].format(answer=example["answer"])
        ans_enc = tok(formatted_answer, truncation=True, add_special_tokens=False, max_length=max_label_len, return_tensors="np")
        answer = ans_enc["input_ids"].tolist()[0]
        
        label = [-100] * max_len
        label[-len(answer):] = answer[-max_len:]
        processed_example["labels"] = label  # Assign 1D list directly
    
    return processed_example


# ─────────────────────────── metrics & helpers ────────────────────────

def find_answer_token_positions(tokenizer, prompt_str, answer_str, input_ids_list, max_len):
    """
    Robustly find the token span in input_ids_list corresponding to answer_str in prompt_str.
    Uses offset mapping if available, otherwise falls back to best-effort substring search.
    Uses the same tokenizer options as main prompt tokenization: padding="max_length", truncation=True, max_length=max_len, return_tensors="np".
    Returns (start_idx, end_idx) or None if not found.
    Args:
        tokenizer: The tokenizer object.
        prompt_str: The full prompt string.
        answer_str: The answer string to locate.
        input_ids_list: The tokenized input_ids list for the prompt.
        max_len: The max length used for tokenization.
    """
    # Try to get offset mapping
    try:
        enc = tokenizer(
            prompt_str,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="np",
            return_offsets_mapping=True,
            add_special_tokens=False
        )
        offsets = enc.get("offset_mapping")[0]
        if offsets is not None:
            # Find answer substring in prompt
            answer_start = prompt_str.find(answer_str)
            if answer_start == -1:
                # Try to ignore whitespace differences
                import re
                match = re.search(re.escape(answer_str.strip()), prompt_str)
                if match:
                    answer_start = match.start()
                else:
                    return None
            answer_end = answer_start + len(answer_str)
            # Find token indices covering this span
            start_idx = end_idx = None
            for i, (s, e) in enumerate(offsets):
                if s <= answer_start < e:
                    start_idx = i
                if s < answer_end <= e:
                    end_idx = i+1
                    break
            if start_idx is not None and end_idx is not None:
                return (start_idx, end_idx)
            # Fallback: cover all tokens overlapping the span
            indices = [i for i, (s, e) in enumerate(offsets) if not (e <= answer_start or s >= answer_end)]
            if indices:
                return (indices[0], indices[-1]+1)
            return None
    except Exception as e:
        pass
    # Fallback: try to match answer token ids as a subsequence
    answer_enc = tokenizer(
        answer_str,
        add_special_tokens=False
    )
    answer_ids = answer_enc["input_ids"]
    for i in range(len(input_ids_list) - len(answer_ids) + 1):
        if input_ids_list[i : i + len(answer_ids)] == answer_ids:
            return (i, i + len(answer_ids))
    return None

exact_match = evaluate.load("exact_match")

def _norm(s: str) -> str:
    """
    Normalize prediction/label strings for exact-match comparison.
    For multi-answer questions (e.g., all_properties), extracts all answers and joins them with '|'.

    :param s: Input string.
    :return: Normalized string.
    """
    import re
    # Extract all \boxed{...} answers
    #boxed = re.findall(r"\\boxed\{\{?(.*?)\}?\}", s)
    # Extract all <result>...</result> answers
    results = re.findall(r"<extra_100>(.*?)</extra_101>", s)
    #values = boxed + results
    if results:
        # Join all extracted values with '|', strip whitespace and trailing periods
        return '|'.join(v.strip().rstrip('.') for v in results)

    return s.strip().rstrip('.')


def compute_metrics_closure(tokenizer: Any) -> Callable[[Any], Any]:
    """
    Compute metrics closure.

    :param tokenizer: Tokenizer instance.
    :return: Metrics computation function.
    """
    def compute_metrics(eval_preds, compute_result: bool = True):
        """
        Compute metrics. With batch_eval_metrics=True, this function is called per batch and at the end with compute_result=True.
        Accumulates predictions and labels across batches, and only computes/returns metrics when compute_result=True.

        :param eval_preds: Evaluation predictions.
        :param compute_result: Whether to return summary statistics (True at end of eval loop).
        :return: Computed metrics (only when compute_result=True).
        """
        # Use closure variables to accumulate results
        if not hasattr(compute_metrics, "all_preds"):
            compute_metrics.all_preds = []
            compute_metrics.all_labels = []
        preds, labels = eval_preds
        # Move to cpu and convert to numpy if needed
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        if isinstance(preds, tuple):
            preds = preds[0]
        if preds.ndim == 3:
            preds = np.argmax(preds, axis=2)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [_norm(p) for p in decoded_preds]
        decoded_labels = [_norm(l) for l in decoded_labels]
        compute_metrics.all_preds.extend(decoded_preds)
        compute_metrics.all_labels.extend(decoded_labels)
        if compute_result:
            # Compute metrics only on the final call
            exact_m = exact_match.compute(
                predictions=compute_metrics.all_preds, references=compute_metrics.all_labels
            )
            # Reset for next eval
            compute_metrics.all_preds = []
            compute_metrics.all_labels = []
            return exact_m
        else:
            # Return empty dict on intermediate calls
            return {}
    return compute_metrics



def show_examples(raw_ds: Any, preds: Any, n: int = 10) -> None:
    """
    Show examples for a specific question set using the appropriate processor.

    :param raw_ds: Raw dataset.
    :param preds: Predictions.
    :param n: Number of examples to show.
    """
    for i in range(min(len(preds), n)):
        smile = raw_ds["smiles"][i]
        q_str = raw_ds["question_template"][i].format(smiles=smile)
        a_str = raw_ds['answer'][i]
        print(f"Q: {q_str}")
        print(f"A: {a_str}")
        print(f"P: {preds[i]}")
        print("-" * 40  )


def do_generation(max_new_tokens: int, tokenizer: Any, model: Any, data: Any) -> list[str]:
    """
    Perform generation.

    :param max_new_tokens: Maximum number of new tokens.
    :param tokenizer: Tokenizer instance.
    :param model: Model instance.
    :param data: Input data.
    :return: A list of generated prediction strings.
    """
    data.set_format(type="torch", columns=["input_ids", "attention_mask"])
    input_ids = data["input_ids"].to(model.device)
    attention_mask = data["attention_mask"].to(model.device)
    
    # Generate text
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            # num_beams=num_beams,
            # do_sample=False,
            # early_stopping=True
        ).to('cpu').numpy()

    # Decode each sequence in the batch
    decoded_sequences = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    # Strip newline characters from each decoded sequence
    response_texts = [seq.strip("\n") for seq in decoded_sequences]

    return response_texts

class PrintFirstExampleCallback(TrainerCallback):
    def __init__(self, tokenizer, train_dataset):
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.has_printed = False

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.has_printed and len(self.train_dataset) > 0:
            example = self.train_dataset[0]
            input_ids = example['input_ids']
            labels = example['labels']
            try:
                decoded_input = self.tokenizer.decode(input_ids, skip_special_tokens=True)
            except Exception:
                decoded_input = str(input_ids)
            try:
                decoded_labels = self.tokenizer.decode(labels, skip_special_tokens=True)
            except Exception:
                decoded_labels = str(labels)
            print("\n[PrintFirstExampleCallback] Sample input:", decoded_input)
            print("[PrintFirstExampleCallback] Expected output:", decoded_labels)
            self.has_printed = True
