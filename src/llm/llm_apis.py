#!/usr/bin/env python
from __future__ import annotations

from typing import Any, Dict, Sequence, Callable

import evaluate
import logging
import numpy as np
import pyarrow as pa
import re
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


# Processor to handle different question sets
class QuestionSetProcessor:
    """
    Base class for handling answer preparation and example display for a question set.

    :param name: Name of the question set.
    :type name: str
    """
    def __init__(self, name: str="") -> None:
        self.name = name

    def prepare_answers(self, table: pa.Table) -> tuple[dict[str, list[Any]], list[bool]]:
        """
        Prepare answers for the question set from a dataset.

        :param table: Dataset containing SMILES and possibly IUPAC names.
        :type table: pa.Table
        :return: Dictionary mapping property/question names to lists of answers.
        :rtype: Dict[str, Sequence[Any]]
        """
        raise NotImplementedError

class IUPACNamingProcessor(QuestionSetProcessor):
    """
    Processor for the IUPAC naming question set.
    """
    def __init__(self, name: str="iupac_naming") -> None:
        super().__init__(name)

    def prepare_answers(self, table: pa.Table) -> tuple[dict[str, list[Any]], list[bool]]:
        """
        Prepare answers for IUPAC naming (just returns the IUPAC names) and a validity mask.

        :param table: Dataset containing IUPAC names.
        :type table: pa.Table
        :return: Tuple of (answer dict, mask) where mask is True for valid names.
        :rtype: Tuple[Dict[str, Sequence[Any]], List[bool]]
        """
        iupac_list = table.column("iupac").to_pylist()
        mask = [x is not None and str(x).strip() != "" for x in iupac_list]
        return {"iupac_name": iupac_list}, mask


class MolecularPropertiesProcessor(QuestionSetProcessor):
    """
    Processor for the molecular properties question set.
    """
    def __init__(self, name: str="molecular_properties") -> None:
        super().__init__(name)

    def prepare_answers(self, table: pa.Table) -> tuple[dict[str, list[Any]], list[bool]]:
        """
        Prepare answers for molecular properties.
        Returns a mask indicating which rows are valid (all properties present and not None/empty).
        """
        smiles = table.column("smiles").to_pylist()
        answers = calculate_molecular_properties(smiles)
        answers["iupac_name"] = table.column("iupac").to_pylist()
        # Ensure all answers are lists of strings
        for k in answers:
            answers[k] = [str(x) for x in answers[k]]
        n = len(smiles)
        mask = [all(str(answers[k][i]).strip() != "" and answers[k][i] is not None for k in answers) for i in range(n)]
        return answers, mask


class AllPropertiesProcessor(QuestionSetProcessor):
    """
    Processor for the comprehensive 'all_properties' question set.
    """
    def __init__(self, name: str="all_properties") -> None:
        super().__init__(name)

    def prepare_answers(self, table: pa.Table) -> tuple[dict[str, list[Any]], list[bool]]:
        """
        Prepare answers for the comprehensive 'all_properties' question set.
        Returns a mask indicating which rows are valid (all properties present and not None/empty).
        """
        smiles = table.column("smiles").to_pylist()
        answers = calculate_molecular_properties(smiles)
        answers["iupac_name"] = table.column("iupac").to_pylist()
        # Ensure all answers are lists of strings
        for k in answers:
            answers[k] = [str(x) for x in answers[k]]
        n = len(smiles)
        mask = [all(str(answers[k][i]).strip() != "" and answers[k][i] is not None for k in answers) for i in range(n)]
        return answers, mask


def do_evaluate(accelerator: Any, model: Any, dataloader: Any, tokenizer: Any, compute_metrics: Any, max_new_tokens: int, num_examples: int = 100) -> dict:
    """
    Run generation-based evaluation and log exact-match metric.

    :param accelerator: Accelerator instance.
    :type accelerator: Accelerator
    :param model: Model instance.
    :type model: Any
    :param dataloader: DataLoader instance.
    :type dataloader: Any
    :param tokenizer: Tokenizer instance.
    :type tokenizer: Any
    :param compute_metrics: Metrics computation function.
    :type compute_metrics: Any
    :param max_new_tokens: Maximum number of new tokens.
    :type max_new_tokens: int
    :param num_examples: Number of examples to evaluate.
    :type num_examples: int
    :return: Dictionary of evaluation metrics.
    :rtype: dict
    """
    logger = logging.getLogger(__name__)
    model.eval()
    num_processed = 0
    for batch in dataloader:
        batch_size = batch["input_ids"].size(0)
        if num_processed + batch_size > num_examples:
            trim = num_examples - num_processed
            for k in batch:
                batch[k] = batch[k][:trim]
            batch_size = trim
        with torch.no_grad():
            generated = accelerator.unwrap_model(model).generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
            )
        generated_padded = accelerator.pad_across_processes(
                generated, dim=1, pad_index=tokenizer.pad_token_id)
        labels_padded = accelerator.pad_across_processes(
                batch["labels"], dim=1, pad_index=-100)
        gen_all    = accelerator.gather(generated_padded)
        labels_all = accelerator.gather(labels_padded)
        compute_metrics((gen_all.cpu().numpy(), labels_all.cpu().numpy()), compute_result=False)
        num_processed += batch_size
        if num_processed >= num_examples:
            break
    metrics = compute_metrics((torch.empty(0), torch.empty(0)), compute_result=True)
    if accelerator.is_main_process:
        try:
            sample_ds = dataloader.dataset.select(range(min(num_examples, len(dataloader.dataset))))
            preds = do_generation(
                max_new_tokens,
                tokenizer,
                accelerator.unwrap_model(model).eval(),
                sample_ds,
            )
            sample_ds.set_format(type="torch", columns=["labels"])
            labels_tensor = sample_ds["labels"].masked_fill(sample_ds["labels"] == -100, tokenizer.pad_token_id)
            gold = tokenizer.batch_decode(labels_tensor, skip_special_tokens=True)
            for i, (g_pred, g_gold) in enumerate(zip(preds, gold)):
                logger.info("\nEXAMPLE %d \n PRED: %s \n GOLD: %s\n", i, g_pred, g_gold)
        except Exception as e:
            logger.warning("Failed to generate example predictions: %s", e)
    model.train()
    return metrics

# =================================================
# Molecular Property Functions
# =================================================
def count_heavy_atoms(mol: Any) -> int:
    """
    Count the number of heavy (non-hydrogen) atoms in a molecule.

    :param mol: RDKit molecule object.
    :type mol: Any
    :return: Number of heavy atoms in the molecule.
    :rtype: int
    """
    if mol is None:
        return 0
    return mol.GetNumHeavyAtoms()

def count_non_hydrogen_bonds(mol: Any) -> int:
    """
    Count the number of bonds not involving hydrogen in a molecule.

    :param mol: RDKit molecule object.
    :type mol: Any
    :return: Number of non-hydrogen bonds in the molecule.
    :rtype: int
    """
    if mol is None:
        return 0
    return sum(
        1
        for bond in mol.GetBonds()
        if bond.GetBeginAtom().GetSymbol() != "H" and bond.GetEndAtom().GetSymbol() != "H"
    )

def count_positive_formal_charge_atoms(mol: Any) -> int:
    """
    Count the number of atoms with positive formal charge in a molecule.

    :param mol: RDKit molecule object.
    :type smiles: str
    :return: Number of atoms with positive formal charge.
    :rtype: int
    """
    if mol is None:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetFormalCharge() > 0)

def count_negative_formal_charge_atoms(mol: Any) -> int:
    """
    Count the number of atoms with negative formal charge in a molecule.

    :param mol: RDKit molecule object.
    :type smiles: str
    :return: Number of atoms with negative formal charge.
    :rtype: int
    """
    if mol is None:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetFormalCharge() < 0)

def count_element_atoms(mol: Any, element: str) -> int:
    """
    Count the number of atoms of a specific element in a molecule.

    :param mol: RDKit molecule object.
    :type mol: Any
    :param element: Chemical symbol of the element to count.
    :type element: str
    :return: Number of atoms of the given element in the molecule.
    :rtype: int
    """
    return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == element)

def count_carbon_atoms(mol: Any) -> int:
    """
    Count carbon atoms in a molecule.

    :param mol: RDKit molecule object.
    :type smiles: str
    :return: Number of carbon atoms in the molecule.
    :rtype: int
    """
    if mol is None:
        return 0
    return count_element_atoms(mol, 'C')

def count_nitrogen_atoms(mol: Any) -> int:
    """
    Count nitrogen atoms in a molecule.

    :param mol: RDKit molecule object.
    :type smiles: str
    :return: Number of nitrogen atoms in the molecule.
    :rtype: int
    """
    if mol is None:
        return 0
    return count_element_atoms(mol, 'N')

def count_oxygen_atoms(mol: Any) -> int:
    """
    Count oxygen atoms in a molecule.

    :param mol: RDKit molecule object.
    :type smiles: str
    :return: Number of oxygen atoms in the molecule.
    :rtype: int
    """
    if mol is None:
        return 0
    return count_element_atoms(mol, 'O')

def count_sulfur_atoms(mol: Any) -> int:
    """
    Count sulfur atoms in a molecule.

    :param mol: RDKit molecule object.
    :type smiles: str
    :return: Number of sulfur atoms in the molecule.
    :rtype: int
    """
    if mol is None:
        return 0
    return count_element_atoms(mol, 'S')

def count_phosphorus_atoms(mol: Any) -> int:
    """
    Count phosphorus atoms in a molecule.

    :param mol: RDKit molecule object.
    :type smiles: str
    :return: Number of phosphorus atoms in the molecule.
    :rtype: int
    """
    if mol is None:
        return 0
    return count_element_atoms(mol, 'P')

def count_chlorine_atoms(mol: Any) -> int:
    """
    Count chlorine atoms in a molecule.

    :param mol: RDKit molecule object.
    :type smiles: str
    :return: Number of chlorine atoms in the molecule.
    :rtype: int
    """
    if mol is None:
        return 0
    return count_element_atoms(mol, 'Cl')

def count_fluorine_atoms(mol: Any) -> int:
    """
    Count fluorine atoms in a molecule.

    :param mol: RDKit molecule object.
    :type smiles: str
    :return: Number of fluorine atoms in the molecule.
    :rtype: int
    """
    if mol is None:
        return 0
    return count_element_atoms(mol, 'F')

def count_rings(mol: Any) -> int:
    """
    Count the number of rings in a molecule.

    :param mol: RDKit molecule object.
    :type smiles: str
    :return: Number of rings in the molecule.
    :rtype: int
    """
    if mol is None:
        return 0
    return rdMolDescriptors.CalcNumRings(mol)

def count_aromatic_rings(mol: Any) -> int:
    """
    Count the number of aromatic rings in a molecule.

    :param mol: RDKit molecule object.
    :type smiles: str
    :return: Number of aromatic rings in the molecule.
    :rtype: int
    """
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

def count_double_bonds(mol: Any) -> int:
    """
    Count the number of double bonds in a molecule.

    :param mol: RDKit molecule object.
    :type smiles: str
    :return: Number of double bonds in the molecule.
    :rtype: int
    """
    if mol is None:
        return 0
    return sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE)

def count_triple_bonds(mol: Any) -> int:
    """
    Count the number of triple bonds in a molecule.

    :param mol: RDKit molecule object.
    :type smiles: str
    :return: Number of triple bonds in the molecule.
    :rtype: int
    """
    if mol is None:
        return 0
    return sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.TRIPLE)

def count_stereo_double_bonds(mol: Any) -> int:
    """
    Count the number of stereo double bonds (E/Z) in a molecule.

    :param mol: RDKit molecule object.
    :type smiles: str
    :return: Number of stereo (E/Z) double bonds in the molecule.
    :rtype: int
    """
    if mol is None:
        return 0
    return sum(1 for bond in mol.GetBonds() 
               if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE 
               and (bond.GetStereo() == Chem.rdchem.BondStereo.STEREOE or 
                    bond.GetStereo() == Chem.rdchem.BondStereo.STEREOZ))

def count_stereocenters(mol: Any) -> int:
    """
    Count the number of stereocenters in a molecule.

    :param mol: RDKit molecule object.
    :type smiles: str
    :return: Number of stereocenters in the molecule.
    :rtype: int
    """
    if mol is None:
        return 0
    return rdMolDescriptors.CalcNumAtomStereoCenters(mol)


# Additional molecular topology functions

def count_five_membered_rings(mol: Any) -> int:
    """Count 5-membered rings in a molecule."""
    if mol is None:
        return 0
    return sum(1 for ring in Chem.GetSymmSSSR(mol) if len(ring) == 5)

def count_aromatic_five_membered_rings(mol: Any) -> int:
    """Count aromatic 5-membered rings in a molecule."""
    if mol is None:
        return 0
    count = 0
    for ring in Chem.GetSymmSSSR(mol):
        atoms = list(ring)
        if len(atoms) != 5:
            continue
        if all(
            mol.GetBondBetweenAtoms(a1, a2).GetIsAromatic()
            for a1, a2 in zip(atoms, atoms[1:] + [atoms[0]])
        ):
            count += 1
    return count

def count_six_membered_rings(mol: Any) -> int:
    """Count 6-membered rings in a molecule."""
    if mol is None:
        return 0
    return sum(1 for ring in Chem.GetSymmSSSR(mol) if len(ring) == 6)

def count_aromatic_six_membered_rings(mol: Any) -> int:
    """Count aromatic 6-membered rings in a molecule."""
    if mol is None:
        return 0
    count = 0
    for ring in Chem.GetSymmSSSR(mol):
        atoms = list(ring)
        if len(atoms) != 6:
            continue
        if all(
            mol.GetBondBetweenAtoms(a1, a2).GetIsAromatic()
            for a1, a2 in zip(atoms, atoms[1:] + [atoms[0]])
        ):
            count += 1
    return count

def longest_chain_length(mol: Any) -> int:
    """Return the length of the longest carbon chain in the molecule where none of the carbons are in a ring."""
    if mol is None:
        return 0
    ri = mol.GetRingInfo()
    ring_atoms = set()
    for ring in ri.AtomRings():
        ring_atoms.update(ring)

    def is_carbon(atom):
        return atom.GetSymbol() == 'C'

    def dfs(atom_idx, visited_atoms, visited_bonds):
        max_len = 1
        atom = mol.GetAtomWithIdx(atom_idx)
        for bond in atom.GetBonds():
            bidx = bond.GetIdx()
            if bidx in visited_bonds:
                continue
            nbr = bond.GetOtherAtomIdx(atom_idx)
            nbr_atom = mol.GetAtomWithIdx(nbr)
            # Only consider carbon atoms that are not in rings
            if not is_carbon(nbr_atom) or nbr in ring_atoms:
                continue
            if nbr in visited_atoms:
                continue
            new_visited_atoms = visited_atoms | {nbr}
            new_visited_bonds = visited_bonds | {bidx}
            max_len = max(max_len, 1 + dfs(nbr, new_visited_atoms, new_visited_bonds))
        return max_len

    overall_max = 0
    for atom in mol.GetAtoms():
        # Only start from carbon atoms that are not in rings
        if not is_carbon(atom) or atom.GetIdx() in ring_atoms:
            continue
        overall_max = max(overall_max, dfs(atom.GetIdx(), {atom.GetIdx()}, set()))
    return overall_max

def count_total_hydrogens(mol: Any) -> int:
    """Count total (implicit + explicit) hydrogens in the molecule."""
    if mol is None:
        return 0
    return sum(atom.GetTotalNumHs() for atom in mol.GetAtoms())

def count_fused_rings(mol: Any) -> int:
    """Return the number of rings in the SSSR that are fused to another ring (using RDKit's IsRingFused)."""
    if mol is None:
        return 0
    ri = mol.GetRingInfo()
    return sum(ri.IsRingFused(i) for i in range(len(ri.AtomRings())))

def count_aromatic_heterocycles(mol: Any) -> int:
    """Count the number of aromatic heterocyclic rings in the molecule."""
    if mol is None:
        return 0
    return rdMolDescriptors.CalcNumAromaticHeterocycles(mol)

def count_aromatic_carbocycles(mol: Any) -> int:
    """Count the number of aromatic carbocyclic rings in the molecule."""
    if mol is None:
        return 0
    return rdMolDescriptors.CalcNumAromaticCarbocycles(mol)

def count_saturated_heterocycles(mol: Any) -> int:
    """Count the number of saturated heterocyclic rings in the molecule."""
    if mol is None:
        return 0
    return rdMolDescriptors.CalcNumSaturatedHeterocycles(mol)

def count_saturated_carbocycles(mol: Any) -> int:
    """Count the number of saturated carbocyclic rings in the molecule."""
    if mol is None:
        return 0
    return rdMolDescriptors.CalcNumSaturatedCarbocycles(mol)

def count_aliphatic_heterocycles(mol: Any) -> int:
    """Count the number of aliphatic heterocyclic rings in the molecule."""
    if mol is None:
        return 0
    return rdMolDescriptors.CalcNumAliphaticHeterocycles(mol)

def count_aliphatic_carbocycles(mol: Any) -> int:
    """Count the number of aliphatic carbocyclic rings in the molecule."""
    if mol is None:
        return 0
    return rdMolDescriptors.CalcNumAliphaticCarbocycles(mol)

# Function to calculate all properties for a set of molecules

def calculate_molecular_properties(smiles_list: Sequence[str]) -> dict[str, list[Any]]:
    """
    Calculate various molecular properties for a list of SMILES strings.

    :return: Dictionary mapping property names to lists of property values.
    :rtype: MutableMapping[str, Sequence[Any]]
    """
    properties = {
        "carbon_count": [],
        "nitrogen_count": [],
        "oxygen_count": [],
        "sulfur_count": [],
        "phosphorus_count": [],
        "chlorine_count": [],
        "fluorine_count": [],
        "ring_count": [],
        "aromatic_ring_count": [],
        "five_membered_ring_count": [],
        "aromatic_five_membered_ring_count": [],
        "six_membered_ring_count": [],
        "aromatic_six_membered_ring_count": [],
        "aromatic_heterocycle_count": [],
        "aromatic_carbocycle_count": [],
        "aliphatic_heterocycle_count": [],
        "aliphatic_carbocycle_count": [],
        "saturated_heterocycle_count": [],
        "saturated_carbocycle_count": [],
        "longest_chain_length": [],
        "hydrogen_count": [],
        "fused_ring_count": [],
        "double_bond_count": [],
        "triple_bond_count": [],
        "stereo_double_bond_count": [],
        "stereocenter_count": [],
        "heavy_atom_count": [],
        "non_hydrogen_bond_count": [],
        "positive_formal_charge_count": [],
        "negative_formal_charge_count": [],
    }

    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        properties["carbon_count"].append(count_carbon_atoms(mol))
        properties["nitrogen_count"].append(count_nitrogen_atoms(mol))
        properties["oxygen_count"].append(count_oxygen_atoms(mol))
        properties["sulfur_count"].append(count_sulfur_atoms(mol))
        properties["phosphorus_count"].append(count_phosphorus_atoms(mol))
        properties["chlorine_count"].append(count_chlorine_atoms(mol))
        properties["fluorine_count"].append(count_fluorine_atoms(mol))
        properties["ring_count"].append(count_rings(mol))
        properties["aromatic_ring_count"].append(count_aromatic_rings(mol))
        properties["five_membered_ring_count"].append(count_five_membered_rings(mol))
        properties["aromatic_five_membered_ring_count"].append(count_aromatic_five_membered_rings(mol))
        properties["six_membered_ring_count"].append(count_six_membered_rings(mol))
        properties["aromatic_six_membered_ring_count"].append(count_aromatic_six_membered_rings(mol))
        properties["longest_chain_length"].append(longest_chain_length(mol))
        properties["hydrogen_count"].append(count_total_hydrogens(mol))
        properties["fused_ring_count"].append(count_fused_rings(mol))
        properties["aromatic_heterocycle_count"].append(count_aromatic_heterocycles(mol))
        properties["aromatic_carbocycle_count"].append(count_aromatic_carbocycles(mol))
        properties["aliphatic_heterocycle_count"].append(count_aliphatic_heterocycles(mol))
        properties["aliphatic_carbocycle_count"].append(count_aliphatic_carbocycles(mol))
        properties["saturated_heterocycle_count"].append(count_saturated_heterocycles(mol))
        properties["saturated_carbocycle_count"].append(count_saturated_carbocycles(mol))
        properties["double_bond_count"].append(count_double_bonds(mol))
        properties["triple_bond_count"].append(count_triple_bonds(mol))
        properties["stereo_double_bond_count"].append(count_stereo_double_bonds(mol))
        properties["stereocenter_count"].append(count_stereocenters(mol))
        properties["heavy_atom_count"].append(count_heavy_atoms(mol))
        properties["non_hydrogen_bond_count"].append(count_non_hydrogen_bonds(mol))
        properties["positive_formal_charge_count"].append(count_positive_formal_charge_atoms(mol))
        properties["negative_formal_charge_count"].append(count_negative_formal_charge_atoms(mol))

    return properties

def process_single_qa(
    tok: Any,
    example: Dict[str, Any],
    max_len: int,
    max_label_len: int | None = None,
    is_train: bool = True,
    system_prompt_override: str | None = None,
) -> Dict[str, Any]:
    """
    Process a single question-answer pair from the expanded dataset.

    :param tok: Tokenizer instance
    :type tok: Any
    :param example: Dictionary containing a single Q&A pair with all necessary fields
    :type example: Dict[str, Any]
    :param max_len: Maximum length for the input
    :type max_len: int
    :param max_label_len: Maximum length for the label (only used for eval)
    :type max_label_len: int or None
    :param is_train: Whether this is for training or evaluation
    :type is_train: bool
    :param system_prompt_override: Optional system prompt string. If provided, this is used instead of ``example['system_prompt']`` (which may be absent when loading datasets created without the system_prompt column).
    :type system_prompt_override: str or None
    :return: Dictionary with tokenized input_ids, attention_mask, and labels
    :rtype: Dict[str, Any]
    """
    # Build the prompt
    if is_train:
        # For training, we include both question and answer
        if hasattr(tok, 'apply_chat_template'):
            # Use chat template if available
            prompt = [
                {"role": "system", "content": system_prompt_override if system_prompt_override is not None else example["system_prompt"]},
                {"role": "user", "content": example["question_template"].format(**example['metadata'])},
                {"role": "assistant", "content": example["assistant_template"].format(**example['metadata'])}
            ]
            prompt_str = tok.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
        else:
            # Fallback for models without chat templates
            system = system_prompt_override if system_prompt_override is not None else example["system_prompt"]
            question = example["question_template"].format(**example['metadata'])
            answer = example["assistant_template"].format(**example['metadata'])
            prompt_str = f"{system}\n\nuser: {question}\n\nassistant: {answer}"
    else:
        # For evaluation, only include the question (no answer)
        if hasattr(tok, 'apply_chat_template'):
            prompt = [  
                {"role": "system", "content": system_prompt_override if system_prompt_override is not None else example.get("system_prompt", "")},
                {"role": "user", "content": example["question_template"].format(**example['metadata'])}
            ]
            prompt_str = tok.apply_chat_template(prompt, add_generation_prompt=True, 
                                                tokenize=False, enable_thinking=False)
        else:
            system = system_prompt_override if system_prompt_override is not None else example.get("system_prompt", "")
            question = example["question_template"].format(**example['metadata'])
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
        answer_text = str(example["assistant_template"].format(**example['metadata']))
        
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
        formatted_answer = example["assistant_template"].format(**example['metadata'])
        ans_enc = tok(formatted_answer, truncation=True, add_special_tokens=False, max_length=max_label_len, return_tensors="np")
        answer = ans_enc["input_ids"].tolist()[0]
        
        label = [-100] * max_len
        label[-len(answer):] = answer[-max_len:]
        processed_example["labels"] = label  # Assign 1D list directly
    
    return processed_example


# ─────────────────────────── metrics & helpers ────────────────────────

def find_answer_token_positions(tokenizer: Any, prompt_str: str, answer_str: str, input_ids_list: list, max_len: int) -> tuple | None:
    """
    Robustly find the token span in input_ids_list corresponding to answer_str in prompt_str.
    Uses offset mapping if available, otherwise falls back to best-effort substring search.
    Uses the same tokenizer options as main prompt tokenization: padding="max_length", truncation=True, max_length=max_len, return_tensors="np".

    :param tokenizer: The tokenizer object.
    :type tokenizer: Any
    :param prompt_str: The full prompt string.
    :type prompt_str: str
    :param answer_str: The answer string to locate.
    :type answer_str: str
    :param input_ids_list: The tokenized input_ids list for the prompt.
    :type input_ids_list: list
    :param max_len: The max length used for tokenization.
    :type max_len: int
    :return: (start_idx, end_idx) or None if not found.
    :rtype: tuple or None
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
    For multi-answer questions (e.g., all_properties), extracts the answer after the last colon and space, up to the next whitespace or '<'.

    :param s: Input string.
    :return: Normalized string.
    """
    # Find the last colon followed by a space
    idx = s.rfind(': ')
    if idx != -1:
        substr = s[idx + 2:]
        # Extract up to the next whitespace or '<'
        import re
        match = re.match(r"([^\s<]+)", substr)
        if match:
            return match.group(1).strip().rstrip('.')
        return substr.strip().split()[0] if substr.strip() else ''
    return s.strip().rstrip('.')


def compute_metrics_closure(tokenizer: Any) -> Callable[[Any], Any]:
    """
    Compute metrics closure.

    :param tokenizer: Tokenizer instance.
    :return: Metrics computation function.
    """
    all_preds = []
    all_labels = []
    def compute_metrics(eval_preds, compute_result: bool = True) -> dict:
        """
        Compute metrics. With batch_eval_metrics=True, this function is called per batch and at the end with compute_result=True.
        Accumulates predictions and labels across batches, and only computes/returns metrics when compute_result=True.

        :param eval_preds: Evaluation predictions.
        :param compute_result: Whether to return summary statistics (True at end of eval loop).
        :return: Computed metrics (only when compute_result=True).
        """
        nonlocal all_preds, all_labels
        preds, labels = eval_preds
        # Move to cpu and convert to numpy if needed
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        if isinstance(preds, tuple):
            preds = preds[0]

        try:
            def filter_out_of_range(tokens, vocab_size, pad_token_id):
                # Replace out-of-range tokens with pad_token_id
                return [t if 0 <= t < vocab_size else pad_token_id for t in tokens]
            # Filter out-of-range tokens for preds and labels
            filtered_preds = [filter_out_of_range(seq, tokenizer.vocab_size, tokenizer.pad_token_id) for seq in preds]
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            filtered_labels = [filter_out_of_range(seq, tokenizer.vocab_size, tokenizer.pad_token_id) for seq in labels]
            decoded_preds = tokenizer.batch_decode(filtered_preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(filtered_labels, skip_special_tokens=True)
            decoded_preds = [_norm(p) for p in decoded_preds]
            decoded_labels = [_norm(label) for label in decoded_labels]
            all_preds.extend(decoded_preds)
            all_labels.extend(decoded_labels)
        except OverflowError:
            print(f"OverflowError: {preds}")
        
        if compute_result:
            # Compute metrics only on the final call
            exact_m = exact_match.compute(
                predictions=all_preds, references=all_labels
            )
            # Reset for next eval
            all_preds = []
            all_labels = []
            return exact_m if exact_m is not None else {}
        else:
            # Return empty dict on intermediate calls
            return {}
    return compute_metrics

def do_generation(max_new_tokens: int, tokenizer: Any, model: Any, data: Any) -> list[str]:
    """
    Perform generation.  Will not work with distributed training.

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

