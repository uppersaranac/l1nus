"""StructureProcessor implementation for structure/isomer multiple-choice questions.

This module defines a `StructureProcessor` subclass of `QuestionSetProcessor`. It
creates new answer columns (`smiles_A`–`smiles_D` and `isomer_answer`) required
by `configs/structure.yaml`. See that YAML file for template details.
"""
from __future__ import annotations

from typing import Dict, Sequence, List
import random

import pyarrow as pa
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from llm.llm_apis import QuestionSetProcessor

__all__ = [
    "StructureProcessor",
]


class StructureProcessor(QuestionSetProcessor):
    """Generate multiple-choice structural isomer questions.

    For each input record we:
        1. Identify the molecule's formula and InChIKey connectivity block.
        2. Sample **three** other records with the *same formula* but a *different* connectivity
           block (structural isomers).
        3. Produce a randomised SMILES for the **correct** answer using
           ``Chem.MolToSmiles(mol, doRandom=True)`` ensuring it differs from the
           original SMILES string.
        4. Shuffle the four choices (A–D) and record which letter contains the
           correct SMILES.
    """

    def __init__(self) -> None:
        super().__init__("structure")

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _noncanonical_smiles(mol: Chem.Mol, original: str) -> str:
        """Return a *randomised* SMILES that is not exactly the original string."""
        for _ in range(12):  # a few attempts should suffice
            rnd = Chem.MolToSmiles(mol, doRandom=True)
            if rnd != original:
                return rnd
        # Fallback (unlikely) – return canonical representation
        return Chem.MolToSmiles(mol, canonical=True)

    # ---------------------------------------------------------------------
    # Main public API
    # ---------------------------------------------------------------------
    def prepare_answers(self, table: pa.Table) -> tuple[Dict[str, Sequence[str]], list[bool]]:
        smiles_col: List[str] = table.column("smiles").to_pylist()
        n = len(smiles_col)

        # Pre-compute RDKit molecules and derived properties
        formulas_all = table.column("formula").to_pylist()
        mols = []
        formulas = []
        key_prefixes = []
        smiles_filtered = []
        for s, f in zip(smiles_col, formulas_all):
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                mols.append(None)
                formulas.append(None)
                key_prefixes.append(None)
                smiles_filtered.append(None)
                continue
            try:
                key_prefix = Chem.MolToInchiKey(mol).split("-")[0]
            except Exception:
                mols.append(None)
                formulas.append(None)
                key_prefixes.append(None)
                smiles_filtered.append(None)
                continue
            mols.append(mol)
            formulas.append(f)
            key_prefixes.append(key_prefix)
            smiles_filtered.append(s)
        # If needed, replace smiles_col with smiles_filtered below for further processing

        # Build mapping formula → list[indices]
        formula_to_indices: Dict[str, List[int]] = {}
        for idx, formula in enumerate(formulas):
            if formula is None:
                continue
            formula_to_indices.setdefault(formula, []).append(idx)

        # Prepare output arrays
        smiles_A: List[str] = []
        smiles_B: List[str] = []
        smiles_C: List[str] = []
        smiles_D: List[str] = []
        isomer_answer: List[str] = []
        mask: List[bool] = []

        letters = "ABCD"
        rng = random.Random()

        for i in range(n):

            if mols[i] is None:
                smiles_A.append("")
                smiles_B.append("")
                smiles_C.append("")
                smiles_D.append("")
                isomer_answer.append("")
                mask.append(False)
                continue

            # Candidate indices: same formula, different connectivity block
            candidates = [j for j in formula_to_indices.get(formulas[i], []) if j != i and key_prefixes[j] != key_prefixes[i]]

            if not candidates:
                smiles_A.append("")
                smiles_B.append("")
                smiles_C.append("")
                smiles_D.append("")
                isomer_answer.append("")
                mask.append(False)
                continue

            # If <3 unique candidates, sample with replacement to reach 3
            if len(candidates) >= 3:
                decoy_indices = rng.sample(candidates, 3)
                decoy_smiles = [smiles_col[j] for j in decoy_indices]
            else:
                # Use all available candidates first
                decoy_indices = list(candidates)
                decoy_smiles = [smiles_col[j] for j in decoy_indices]
                # Generate additional decoys using _noncanonical_smiles
                while len(decoy_smiles) < 3:
                    idx = rng.choice(candidates)
                    mol = mols[idx]
                    orig_smiles = smiles_col[idx]
                    new_smiles = self._noncanonical_smiles(mol, orig_smiles)
                    decoy_smiles.append(new_smiles)

            # Randomised correct SMILES representation
            correct_smiles = self._noncanonical_smiles(mols[i], smiles_filtered[i])

            choices = decoy_smiles + [correct_smiles]
            rng.shuffle(choices)
            correct_idx = choices.index(correct_smiles)

            # Append to output lists
            smiles_A.append(choices[0])
            smiles_B.append(choices[1])
            smiles_C.append(choices[2])
            smiles_D.append(choices[3])
            isomer_answer.append(letters[correct_idx])
            mask.append(True)

        return {
            "smiles_A": smiles_A,
            "smiles_B": smiles_B,
            "smiles_C": smiles_C,
            "smiles_D": smiles_D,
            "isomer_answer": isomer_answer,
        }, mask

