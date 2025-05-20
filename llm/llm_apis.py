#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path
import random
from typing import Any, Sequence, Dict, Optional, Callable

import numpy as np
import evaluate
from datasets import Dataset

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
        "system_prompt": SYSTEM_PROMPT,
        "questions": [
            {
                "id": "iupac_name",
                "user_template": "Use the IUPAC naming rules to name the molecule {smiles}.",
                "assistant_template": "The correct IUPAC name for this structure is <result>{answer}</result>."
            }
        ]
    },
    "molecular_properties": {
        "system_prompt": SYSTEM_PROMPT,
        "questions": [
            {
                "id": "carbon_count",
                "user_template": "How many carbon atoms are in the molecule {smiles}?",
                "assistant_template": "\\boxed{{{answer}}}"
            },
            {
                "id": "heavy_atom_count",
                "user_template": "How many heavy (non-hydrogen) atoms are in the molecule {smiles}?",
                "assistant_template": "\\boxed{{{answer}}}"
            },
            {
                "id": "non_hydrogen_bond_count",
                "user_template": "How many bonds not to hydrogen are in the molecule {smiles}?",
                "assistant_template": "\\boxed{{{answer}}}"
            },
            {
                "id": "positive_formal_charge_count",
                "user_template": "How many atoms with positive formal charge are in the molecule {smiles}?",
                "assistant_template": "\\boxed{{{answer}}}"
            },
            {
                "id": "negative_formal_charge_count",
                "user_template": "How many atoms with negative formal charge are in the molecule {smiles}?",
                "assistant_template": "\\boxed{{{answer}}}"
            },
            {
                "id": "nitrogen_count",
                "user_template": "How many nitrogen atoms are in the molecule {smiles}?",
                "assistant_template": "\\boxed{{{answer}}}"
            },
            {
                "id": "oxygen_count",
                "user_template": "How many oxygen atoms are in the molecule {smiles}?",
                "assistant_template": "\\boxed{{{answer}}}"
            },
            {
                "id": "sulfur_count",
                "user_template": "How many sulfur atoms are in the molecule {smiles}?",
                "assistant_template": "\\boxed{{{answer}}}"
            },
            {
                "id": "phosphorus_count",
                "user_template": "How many phosphorus atoms are in the molecule {smiles}?", 
                "assistant_template": "\\boxed{{{answer}}}"
            },
            {
                "id": "chlorine_count", 
                "user_template": "How many chlorine atoms are in the molecule {smiles}?",
                "assistant_template": "\\boxed{{{answer}}}"
            },
            {
                "id": "fluorine_count",
                "user_template": "How many fluorine atoms are in the molecule {smiles}?",
                "assistant_template": "\\boxed{{{answer}}}"
            },
            {
                "id": "ring_count",
                "user_template": "How many rings are in the molecule {smiles}?",
                "assistant_template": "\\boxed{{{answer}}}"
            },
            {
                "id": "aromatic_ring_count",
                "user_template": "How many aromatic rings are in the molecule {smiles}?",
                "assistant_template": "\\boxed{{{answer}}}"
            },
            {
                "id": "double_bond_count",
                "user_template": "How many double bonds are in the molecule {smiles}?",
                "assistant_template": "\\boxed{{{answer}}}"
            },
            {
                "id": "triple_bond_count",
                "user_template": "How many triple bonds are in the molecule {smiles}?",
                "assistant_template": "\\boxed{{{answer}}}"
            },
            {
                "id": "stereo_double_bond_count",
                "user_template": "How many stereo double bonds are in the molecule {smiles}?",
                "assistant_template": "\\boxed{{{answer}}}"
            },
            {
                "id": "stereocenter_count",
                "user_template": "How many stereocenters are in the molecule {smiles}?",
                "assistant_template": "\\boxed{{{answer}}}"
            }
        ]
    },
    "all_properties": {
        "system_prompt": SYSTEM_PROMPT,
        "questions": [
            {
                "id": "all_properties",
                "user_template": "Analyze the following molecular properties for the molecule {smiles}: carbon atoms, nitrogen atoms, oxygen atoms, sulfur atoms, phosphorus atoms, chlorine atoms, fluorine atoms, rings, aromatic rings, double bonds, triple bonds, stereo double bonds, stereocenters, and the IUPAC name. Provide a comprehensive report.",
                "assistant_template": "Molecular Analysis of {smiles}:\n\nCarbon atoms: \\boxed{{{carbon_count}}}\nNitrogen atoms: \\boxed{{{nitrogen_count}}}\nOxygen atoms: \\boxed{{{oxygen_count}}}\nSulfur atoms: \\boxed{{{sulfur_count}}}\nPhosphorus atoms: \\boxed{{{phosphorus_count}}}\nChlorine atoms: \\boxed{{{chlorine_count}}}\nFluorine atoms: \\boxed{{{fluorine_count}}}\nRings: \\boxed{{{ring_count}}}\nAromatic rings: \\boxed{{{aromatic_ring_count}}}\nDouble bonds: \\boxed{{{double_bond_count}}}\nTriple bonds: \\boxed{{{triple_bond_count}}}\nStereo double bonds: \\boxed{{{stereo_double_bond_count}}}\nStereocenters: \\boxed{{{stereocenter_count}}}\nHeavy atoms: \\boxed{{{heavy_atom_count}}}\nBonds not to hydrogen: \\boxed{{{non_hydrogen_bond_count}}}\nAtoms with positive formal charge: \\boxed{{{positive_formal_charge_count}}}\nAtoms with negative formal charge: \\boxed{{{negative_formal_charge_count}}}\nIUPAC name: <result>{iupac_name}</result>"
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


# ───────────────────────── tokenisation helpers ───────────────────────
def build_train_batch(tok: Any, smiles: Sequence[str], answers: Dict[str, Sequence[str]], max_len: int, question_set_name: str = "iupac_naming") -> Dict[str, Any]:
    """
    Build a training batch for a specific question set.

    :param tok: The tokenizer.
    :param smiles: List of SMILES strings.
    :param answers: Dictionary mapping question ids to lists of answers.
    :param max_len: Maximum sequence length.
    :param question_set_name: The name of the question set to use.
    :return: Encoded batch with input_ids, attention_mask, and labels.
    """
    question_set = QUESTION_SETS[question_set_name]
    msgs = []
    
    for i, s in enumerate(smiles):
        for question in question_set["questions"]:
            qid = question["id"]
            if qid in answers and i < len(answers[qid]):
                a = answers[qid][i]
                msgs.append([
                    {"role": "system", "content": question_set["system_prompt"]},
                    {"role": "user", "content": question["user_template"].format(smiles=s)},
                    {"role": "assistant", "content": question["assistant_template"].format(answer=a)}
                ])
    
    # Get prompt strings
    prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in msgs]
    
    # Tokenize prompts
    enc = tok(prompts, padding="max_length", truncation=True,
              max_length=max_len, return_tensors="np")
    
    # Extract all answers for label creation
    all_answers = []
    for i, s in enumerate(smiles):
        for question in question_set["questions"]:
            qid = question["id"]
            if qid in answers and i < len(answers[qid]):
                all_answers.append(answers[qid][i])
    
    # Prepare answer token IDs without special tokens
    answers_ids = [tok(text, add_special_tokens=False)["input_ids"] for text in all_answers]
    input_ids_list = enc["input_ids"].tolist()
    
    labels_full = []
    for idx, (row_ids, ans_ids) in enumerate(zip(input_ids_list, answers_ids)):
        # Find the start index of the answer tokens within the input_ids sequence
        start_idx = -1
        for i in range(len(row_ids) - len(ans_ids) + 1):
            if row_ids[i : i + len(ans_ids)] == ans_ids[:len(row_ids)-i]:
                start_idx = i
                break
                
        label = [-100] * len(row_ids)
        if start_idx >= 0:
            # Build label row: mask (-100) everywhere except the answer span
            for j, tok_id in enumerate(ans_ids):
                label[start_idx + j] = tok_id
        else:
            print(f"Warning: Answer not found in input_ids for example {idx}")
            
        labels_full.append(label)
    
    enc["labels"] = labels_full
    return enc


def build_eval_batch(tok: Any, smiles: Sequence[str], answers: Dict[str, Sequence[str]], max_prompt_len: int, max_label_len: int, question_set_name: str = "iupac_naming") -> Dict[str, Any]:
    """
    Build an evaluation batch for a specific question set.

    :param tok: The tokenizer.
    :param smiles: List of SMILES strings.
    :param answers: Dictionary mapping question ids to lists of answers.
    :param max_prompt_len: Maximum prompt length.
    :param max_label_len: Maximum label length.
    :param question_set_name: The name of the question set to use.
    :return: Encoded batch with input_ids, attention_mask, and labels.
    """
    question_set = QUESTION_SETS[question_set_name]
    user_msgs = []
    
    for i, s in enumerate(smiles):
        for question in question_set["questions"]:
            qid = question["id"]
            if qid in answers and i < len(answers[qid]):
                user_msgs.append([
                    {"role": "system", "content": question_set["system_prompt"]},
                    {"role": "user", "content": question["user_template"].format(smiles=s)}
                ])
    
    # Prompt strings
    prompts = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=False)
               for m in user_msgs]
    
    # Tokenize prompts
    prompt_enc = tok(prompts, padding="max_length", truncation=True,
                     max_length=max_prompt_len, return_tensors="np")
    
    # Extract all answers for label creation
    all_answers = []
    all_templates = []
    for i, s in enumerate(smiles):
        for question in question_set["questions"]:
            qid = question["id"]
            if qid in answers and i < len(answers[qid]):
                all_answers.append(answers[qid][i])
                all_templates.append(question["assistant_template"])
    
    # Labels: tokenize only the answers
    formatted_answers = [template.format(answer=a) for template, a in zip(all_templates, all_answers)]
    ans_enc = tok(formatted_answers, truncation=True, add_special_tokens=False,
                  max_length=max_label_len, return_tensors="np")
    
    # Build labels matching the full sequence length, ignoring prompt tokens
    answers = ans_enc["input_ids"].tolist()
    labels_full = []
    for answer in answers:
        # Initialize all positions to ignore_index (-100)
        label = [-100] * max_prompt_len
        # Right-align the answer tokens at the end of the sequence
        offset = max_prompt_len - len(answer)
        label[offset:] = answer
        labels_full.append(label)
        
    prompt_enc["labels"] = labels_full
    return prompt_enc


# ─────────────────────────── metrics & helpers ────────────────────────
exact_match = evaluate.load("exact_match")

def _norm(s: str) -> str:
    """
    Normalize prediction/label strings for exact-match comparison, handling \boxed{{}} and <result>...</result> tags.
    For multi-answer questions (e.g., all_properties), extracts all answers and joins them with '|'.

    :param s: Input string.
    :return: Normalized string.
    """
    import re
    # Extract all \boxed{...} answers
    boxed = re.findall(r"\\boxed\{\{?(.*?)\}?\}", s)
    # Extract all <result>...</result> answers
    results = re.findall(r"<result>(.*?)</result>", s)
    values = boxed + results
    if values:
        # Join all extracted values with '|', strip whitespace and trailing periods
        return '|'.join(v.strip().rstrip('.') for v in values)

    return s.strip().rstrip('.')


def compute_metrics_closure(tokenizer: Any) -> Callable[[Any], Any]:
    """
    Compute metrics closure.

    :param tokenizer: Tokenizer instance.
    :return: Metrics computation function.
    """
    def compute_metrics(eval_preds):
        """
        Compute metrics.

        :param eval_preds: Evaluation predictions.
        :return: Computed metrics.
        """
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Simple normalization
        decoded_preds = [_norm(p) for p in decoded_preds]
        decoded_labels = [_norm(l) for l in decoded_labels]
        
        # Compute exact match
        exact_m = exact_match.compute(
            predictions=decoded_preds, references=decoded_labels
        )
        
        return exact_m
        
    return compute_metrics


def show_examples(raw_ds: Dataset, preds: Any, tok: Any, n: int = 10) -> None:
    """
    Show examples.

    :param raw_ds: Raw dataset.
    :param preds: Predictions.
    :param tok: Tokenizer.
    :param n: Number of examples to show.
    """
    # Show some examples
    for i in range(0, len(raw_ds), len(raw_ds)//n):
        q = f"What is the IUPAC name for the molecule {raw_ds[i]['smiles']}?"
        gt = f"It is {raw_ds[i]['iupac']}"
        
        # Decode the prediction
        pd = tok.decode(preds[i])
        
        # Collapse consecutive  tokens into a single token
        import re
        pd = re.sub(r'(<\|im_end\|>\s*)+', ' ', pd)
        
        print(f"\n#{i}")
        print("Q :", q)
        print("GT:", gt)
        print("PD:", pd)


def do_generation(num_beams: int, max_new_tokens: int, tokenizer: Any, model: Any, data: Any) -> np.ndarray:
    """
    Perform generation.

    :param num_beams: Number of beams.
    :param max_new_tokens: Maximum number of new tokens.
    :param tokenizer: Tokenizer instance.
    :param model: Model instance.
    :param data: Input data.
    :return: Generated predictions.
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
            num_beams=num_beams,
            do_sample=False,
            early_stopping=True
        ).to('cpu').numpy()

    # Strip the input_ids from the generated_ids
    seq_lens = attention_mask.sum(dim=1).to('cpu').numpy()
    preds = [g[s:] for g, s in zip(generated_ids, seq_lens)]

    # pad the predictions
    max_pred_len = max(len(p) for p in preds)
    padded_preds = np.zeros((len(preds), max_pred_len), dtype=np.int64)
    for i, p in enumerate(preds):
        padded_preds[i, :len(p)] = p
    
    return padded_preds
