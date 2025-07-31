"""
llm_mol.py: RDKit-dependent molecular property functions for l1nus

All functions in this file are documented and use type hints.
"""
from typing import Any, Sequence
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def count_heavy_atoms(mol: Any) -> int:
    """
    Count the number of heavy (non-hydrogen) atoms in a molecule.
    """
    if mol is None:
        return 0
    return mol.GetNumHeavyAtoms()

def count_non_hydrogen_bonds(mol: Any) -> int:
    """
    Count the number of bonds not involving hydrogen in a molecule.
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
    """
    if mol is None:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetFormalCharge() > 0)

def count_negative_formal_charge_atoms(mol: Any) -> int:
    """
    Count the number of atoms with negative formal charge in a molecule.
    """
    if mol is None:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetFormalCharge() < 0)

def count_element_atoms(mol: Any, element: str) -> int:
    """
    Count the number of atoms of a specific element in a molecule.
    """
    return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == element)

def count_carbon_atoms(mol: Any) -> int:
    """
    Count carbon atoms in a molecule.
    """
    if mol is None:
        return 0
    return count_element_atoms(mol, 'C')

def count_nitrogen_atoms(mol: Any) -> int:
    """
    Count nitrogen atoms in a molecule.
    """
    if mol is None:
        return 0
    return count_element_atoms(mol, 'N')

def count_oxygen_atoms(mol: Any) -> int:
    """
    Count oxygen atoms in a molecule.
    """
    if mol is None:
        return 0
    return count_element_atoms(mol, 'O')

def count_sulfur_atoms(mol: Any) -> int:
    """
    Count sulfur atoms in a molecule.
    """
    if mol is None:
        return 0
    return count_element_atoms(mol, 'S')

def count_phosphorus_atoms(mol: Any) -> int:
    """
    Count phosphorus atoms in a molecule.
    """
    if mol is None:
        return 0
    return count_element_atoms(mol, 'P')

def count_chlorine_atoms(mol: Any) -> int:
    """
    Count chlorine atoms in a molecule.
    """
    if mol is None:
        return 0
    return count_element_atoms(mol, 'Cl')

def count_fluorine_atoms(mol: Any) -> int:
    """
    Count fluorine atoms in a molecule.
    """
    if mol is None:
        return 0
    return count_element_atoms(mol, 'F')

def count_rings(mol: Any) -> int:
    """
    Count the number of rings in a molecule.
    """
    if mol is None:
        return 0
    return rdMolDescriptors.CalcNumRings(mol)

def count_aromatic_rings(mol: Any) -> int:
    """
    Count the number of aromatic rings in a molecule.
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
    """
    if mol is None:
        return 0
    return sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE)

def count_triple_bonds(mol: Any) -> int:
    """
    Count the number of triple bonds in a molecule.
    """
    if mol is None:
        return 0
    return sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.TRIPLE)

def count_stereo_double_bonds(mol: Any) -> int:
    """
    Count the number of stereo double bonds (E/Z) in a molecule.
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
    """
    if mol is None:
        return 0
    return rdMolDescriptors.CalcNumAtomStereoCenters(mol)

def count_five_membered_rings(mol: Any) -> int:
    """
    Count 5-membered rings in a molecule.
    """
    if mol is None:
        return 0
    return sum(1 for ring in Chem.GetSymmSSSR(mol) if len(ring) == 5)

def count_aromatic_five_membered_rings(mol: Any) -> int:
    """
    Count aromatic 5-membered rings in a molecule.
    """
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
    """
    Count 6-membered rings in a molecule.
    """
    if mol is None:
        return 0
    return sum(1 for ring in Chem.GetSymmSSSR(mol) if len(ring) == 6)

def count_aromatic_six_membered_rings(mol: Any) -> int:
    """
    Count aromatic 6-membered rings in a molecule.
    """
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
    """
    Return the length of the longest carbon chain in the molecule where none of the carbons are in a ring.
    """
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
    """
    Count total (implicit + explicit) hydrogens in the molecule.
    """
    if mol is None:
        return 0
    return sum(atom.GetTotalNumHs() for atom in mol.GetAtoms())

def count_fused_rings(mol: Any) -> int:
    """
    Return the number of rings in the SSSR that are fused to another ring (using RDKit's IsRingFused).
    """
    if mol is None:
        return 0
    ri = mol.GetRingInfo()
    return sum(ri.IsRingFused(i) for i in range(len(ri.AtomRings())))

def count_aromatic_heterocycles(mol: Any) -> int:
    """
    Count the number of aromatic heterocyclic rings in the molecule.
    """
    if mol is None:
        return 0
    return rdMolDescriptors.CalcNumAromaticHeterocycles(mol)

def count_aromatic_carbocycles(mol: Any) -> int:
    """
    Count the number of aromatic carbocyclic rings in the molecule.
    """
    if mol is None:
        return 0
    return rdMolDescriptors.CalcNumAromaticCarbocycles(mol)

def count_saturated_heterocycles(mol: Any) -> int:
    """
    Count the number of saturated heterocyclic rings in the molecule.
    """
    if mol is None:
        return 0
    return rdMolDescriptors.CalcNumSaturatedHeterocycles(mol)

def count_saturated_carbocycles(mol: Any) -> int:
    """
    Count the number of saturated carbocyclic rings in the molecule.
    """
    if mol is None:
        return 0
    return rdMolDescriptors.CalcNumSaturatedCarbocycles(mol)

def count_aliphatic_heterocycles(mol: Any) -> int:
    """
    Count the number of aliphatic heterocyclic rings in the molecule.
    """
    if mol is None:
        return 0
    return rdMolDescriptors.CalcNumAliphaticHeterocycles(mol)

def count_aliphatic_carbocycles(mol: Any) -> int:
    """
    Count the number of aliphatic carbocyclic rings in the molecule.
    """
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

def smiles_with_ring_atom_classes(mol: Any) -> str:
    """
    Return a SMILES string where each atom's atom class is set to the ring index (or indices) if the atom is in a ring.
    If an atom is in multiple rings, the atom class is a concatenation of all ring indices (in increasing order).
    If a ring index is two digits, prepend it with a % sign (e.g., %10).
    Atoms not in any ring have no atom class.
    This is done by first setting atom map numbers to atom index + 1, generating the SMILES, then using regex to replace
    [C:idx] with [C:class] where class is the ring index string as described above.

    :param mol: RDKit molecule object
    :return: SMILES string with atom classes encoding ring membership
    """
    import copy
    import re
    mol = copy.deepcopy(mol)
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()  # tuple of atom idxs for each ring
    # Build a mapping: atom idx -> list of ring indices
    atom_to_rings = {i: [] for i in range(mol.GetNumAtoms())}
    for ring_idx, ring in enumerate(atom_rings):
        for atom_idx in ring:
            atom_to_rings[atom_idx].append(ring_idx)
    # Set atom map number for each atom to atom index + 1
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    # Generate SMILES with atom map numbers
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    # Regex to find [X:idx] and replace with [X:class]
    def ring_class_str(rings):
        s = ''
        for r in sorted(rings):
            ring_label = r + 1  # Use ring index + 1
            if ring_label >= 10:
                s += f'%{ring_label}'
            else:
                s += str(ring_label)
        return s
    def replace_atom(match):
        atom = match.group(1)
        idx = int(match.group(2)) - 1  # atom map idx is atom index + 1
        rings = atom_to_rings.get(idx, [])
        if rings:
            cls = ring_class_str(rings)
            return f'[{atom}:{cls}]'
        else:
            return f'[{atom}]'
    # Replace all [X:idx] with [X:class] or [X] if not in ring
    smiles = re.sub(r'\[([^\]:]+):(\d+)\]', replace_atom, smiles)
    return smiles

