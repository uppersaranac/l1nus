"""
llm_mol.py: RDKit-dependent molecular property functions for l1nus

All functions in this file are documented and use type hints.
"""
from typing import Any, Sequence
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Kekulize, MolToSmiles
import logging
import copy

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

def get_net_formal_charge(mol: Any) -> int:
    """
    Return the net formal charge of the molecule (sum of all formal charges).
    
    :param mol: RDKit molecule object
    :return: Net formal charge as an integer
    """
    if mol is None:
        return 0
    return sum(atom.GetFormalCharge() for atom in mol.GetAtoms())

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

def get_stereo_summary(mol: Any) -> list[int]:
    """
    Return a list: [number of stereocenters, number of stereo bonds] for the molecule.

    :param mol: RDKit molecule object
    :return: [stereocenter_count, stereo_bond_count]
    """
    if mol is None:
        return [0, 0]
    stereocenter_count = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    stereo_bond_count = sum(
        1
        for bond in mol.GetBonds()
        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE and bond.GetStereo() in [Chem.rdchem.BondStereo.STEREOE, Chem.rdchem.BondStereo.STEREOZ]
    )
    return [stereocenter_count, stereo_bond_count]

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

def longest_chain(mol: Any) -> set[int]:
    """
    Return the list of atom indices in the longest carbon chain in the molecule where none of the carbons are in a ring.

    :param mol: RDKit molecule object
    :return: List of atom indices in the longest chain
    """
    if mol is None:
        return set()
    ri = mol.GetRingInfo()
    ring_atoms = set()
    for ring in ri.AtomRings():
        ring_atoms.update(ring)

    def is_carbon(atom):
        return atom.GetSymbol() == 'C'

    def dfs(atom_idx, visited_atoms, visited_bonds, path):
        atom = mol.GetAtomWithIdx(atom_idx)
        longest = list(path)
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
            candidate = dfs(nbr, new_visited_atoms, new_visited_bonds, path + [nbr])
            if len(candidate) > len(longest):
                longest = candidate
        return longest

    longest_chain = []
    for atom in mol.GetAtoms():
        # Only start from carbon atoms that are not in rings
        if not is_carbon(atom) or atom.GetIdx() in ring_atoms:
            continue
        candidate = dfs(atom.GetIdx(), {atom.GetIdx()}, set(), [atom.GetIdx()])
        candidate = [idx + 1 for idx in candidate]  # Convert to 1-based indexing
        if len(candidate) > len(longest_chain):
            longest_chain = candidate
    return set(longest_chain)

def sorted_rings(mol: Any) -> list[set[int]]:
    """
    Return all rings in the molecule as lists of atom indices, sorted in descending order within each ring.
    The rings themselves are sorted by the max atom index (descending), then by min atom index (descending).

    :param mol: RDKit molecule object
    :return: List of rings, each a list of atom indices (sorted descending)
    """
    if mol is None:
        return []
    rings = [set(ring) for ring in mol.GetRingInfo().AtomRings()]
    # Sort rings by max atom index (descending), then min atom index (descending)
    rings.sort(key=lambda r: (max(r), min(r)))
    return rings

def kekulized_smiles(mol: Any, atom_map: bool = False) -> str:
    """
    Return the kekulized SMILES string for the given molecule.

    :param mol: RDKit molecule object
    :param atom_map: If True, output atom map numbers (atom index + 1)
    :return: Kekulized SMILES string
    """
    if mol is None:
        return ""
    mol_kek = copy.deepcopy(mol)
    if atom_map:
        for atom in mol_kek.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
    else:
        for atom in mol_kek.GetAtoms():
            atom.SetAtomMapNum(0)
    try:
        Kekulize(mol_kek, clearAromaticFlags=True)
    except Exception:
        result = MolToSmiles(mol, isomericSmiles=True)
        logging.warning(f"Kekulization failed for molecule {result}, returning non-kekulized SMILES")
        return result
    return MolToSmiles(mol_kek, kekuleSmiles=True, isomericSmiles=True)

def get_hybridization_indices(mol: Any) -> list[set[int]]:
    """
    Return atom indices for sp3, sp2, and sp hybridization in the molecule.
    Each list is sorted in descending order.

    :param mol: RDKit molecule object
    :return: Dict with keys 'sp3', 'sp2', 'sp' and values as lists of atom indices (sorted descending)
    """
    if mol is None:
        return [set(), set(), set()]
    from rdkit.Chem.rdchem import HybridizationType
    sp3 = []
    sp2 = []
    sp = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx() + 1 # Use 1-based indexing for consistency with atom maps
        hyb = atom.GetHybridization()
        if hyb == HybridizationType.SP3:
            sp3.append(idx)
        elif hyb == HybridizationType.SP2:
            sp2.append(idx)
        elif hyb == HybridizationType.SP:
            sp.append(idx)
    return [
        set(sp3),
        set(sp2),
        set(sp)
    ]

def get_element_counts(mol: Any) -> list[Any]:
    """
    Return lists of atom counts for C, N, O, P, S, Cl, F in the molecule.

    :param mol: RDKit molecule object
    :return: List of ints: [C, N, O, P, S, Cl, F]
    """
    if mol is None:
        return [set() for _ in range(7)]
    elements = ['C', 'N', 'O', 'P', 'S', 'Cl', 'F']
    indices = [[] for _ in elements]
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        idx = atom.GetIdx()
        for i, el in enumerate(elements):
            if symbol == el:
                indices[i].append(idx)
    return [len(lst) for lst in indices]

def get_bond_counts(mol: Any) -> list[int]:
    """
    Return a list with total number of bonds (including to hydrogen), number of double bonds, and number of triple bonds.

    :param mol: RDKit molecule object
    :return: [total_bonds, double_bonds, triple_bonds]
    """
    if mol is None:
        return [0, 0, 0]
    mol_kek = copy.deepcopy(mol)
    try:
        Kekulize(mol_kek, clearAromaticFlags=True)
    except Exception:
        logging.warning(f"Kekulization failed for molecule {MolToSmiles(mol)}, returning non-kekulized bond counts")
        pass

    total_bonds = mol_kek.GetNumBonds()
    double_bonds = 0
    triple_bonds = 0
    for bond in mol_kek.GetBonds():
        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            double_bonds += 1
        elif bond.GetBondType() == Chem.rdchem.BondType.TRIPLE:
            triple_bonds += 1
    return [total_bonds, double_bonds, triple_bonds]

def get_ring_counts(mol: Any) -> list[int]:
    """
    Return a list containing the number of: all rings, aromatic rings, 5-membered rings, 6-membered rings.

    :param mol: RDKit molecule object
    :return: [all_rings, aromatic_rings, five_membered_rings, six_membered_rings]
    """
    if mol is None:
        return [0, 0, 0, 0]
    all_rings = rdMolDescriptors.CalcNumRings(mol)
    aromatic_rings = 0
    for ring in Chem.GetSymmSSSR(mol):
        atoms = list(ring)
        if all(mol.GetBondBetweenAtoms(a1, a2).GetIsAromatic() for a1, a2 in zip(atoms, atoms[1:] + [atoms[0]])):
            aromatic_rings += 1
    five_membered_rings = sum(1 for ring in Chem.GetSymmSSSR(mol) if len(ring) == 5)
    six_membered_rings = sum(1 for ring in Chem.GetSymmSSSR(mol) if len(ring) == 6)
    return [all_rings, aromatic_rings, five_membered_rings, six_membered_rings]

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


def get_molecular_formula(mol: Any) -> str:
    """
    Return the chemical formula for the molecule.
    
    :param mol: RDKit molecule object
    :return: Chemical formula as a string (e.g., "C6H12O6")
    """
    if mol is None:
        return ""
    return rdMolDescriptors.CalcMolFormula(mol)


# Function to calculate all properties for a set of molecules

def calculate_molecular_properties(smiles_list: Sequence[str]) -> dict[str, list[Any]]:
    """
    Calculate various molecular properties for a list of SMILES strings.

    :return: Dictionary mapping property names to lists of property values.
    :rtype: MutableMapping[str, Sequence[Any]]
    """
    properties = {
#        "carbon_count": [],
#        "nitrogen_count": [],
#        "oxygen_count": [],
#        "sulfur_count": [],
#        "phosphorus_count": [],
#        "chlorine_count": [],
#        "fluorine_count": [],
#        "ring_count": [],
#        "aromatic_ring_count": [],
#        "five_membered_ring_count": [],
#        "aromatic_five_membered_ring_count": [],
#        "six_membered_ring_count": [],
#        "aromatic_six_membered_ring_count": [],
#        "aromatic_heterocycle_count": [],
#        "aromatic_carbocycle_count": [],
#        "aliphatic_heterocycle_count": [],
#        "aliphatic_carbocycle_count": [],
#        "saturated_heterocycle_count": [],
#        "saturated_carbocycle_count": [],
        "longest_chain": [],
        "hydrogen_count": [],
#        "fused_ring_count": [],
#        "double_bond_count": [],
#        "triple_bond_count": [],
#        "stereo_double_bond_count": [],
        "heavy_atom_count": [],
#        "non_hydrogen_bond_count": [],
#        "positive_formal_charge_count": [],
#        "negative_formal_charge_count": [],
        "net_formal_charge": [],
        # New functions
        "sorted_rings": [],
        "kekulized_smiles": [],
        "kekulized_smiles_atom_map": [],
        "hybridization_indices": [],
        "element_counts": [],
        "bond_counts": [],
        "ring_counts": [],
        "stereo_summary": [],
        "molecular_formula": [],
    }

    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
#        properties["carbon_count"].append(count_carbon_atoms(mol))
#        properties["nitrogen_count"].append(count_nitrogen_atoms(mol))
#        properties["oxygen_count"].append(count_oxygen_atoms(mol))
#        properties["sulfur_count"].append(count_sulfur_atoms(mol))
#        properties["phosphorus_count"].append(count_phosphorus_atoms(mol))
#        properties["chlorine_count"].append(count_chlorine_atoms(mol))
#        properties["fluorine_count"].append(count_fluorine_atoms(mol))
#        properties["ring_count"].append(count_rings(mol))
#        properties["aromatic_ring_count"].append(count_aromatic_rings(mol))
#        properties["five_membered_ring_count"].append(count_five_membered_rings(mol))
#        properties["aromatic_five_membered_ring_count"].append(count_aromatic_five_membered_rings(mol))
#        properties["six_membered_ring_count"].append(count_six_membered_rings(mol))
#        properties["aromatic_six_membered_ring_count"].append(count_aromatic_six_membered_rings(mol))
        properties["longest_chain"].append(longest_chain(mol))
        properties["hydrogen_count"].append(count_total_hydrogens(mol))
#        properties["fused_ring_count"].append(count_fused_rings(mol))
#        properties["aromatic_heterocycle_count"].append(count_aromatic_heterocycles(mol))
#        properties["aromatic_carbocycle_count"].append(count_aromatic_carbocycles(mol))
#        properties["aliphatic_heterocycle_count"].append(count_aliphatic_heterocycles(mol))
#        properties["aliphatic_carbocycle_count"].append(count_aliphatic_carbocycles(mol))
#        properties["saturated_heterocycle_count"].append(count_saturated_heterocycles(mol))
#        properties["saturated_carbocycle_count"].append(count_saturated_carbocycles(mol))
#        properties["double_bond_count"].append(count_double_bonds(mol))
#        properties["triple_bond_count"].append(count_triple_bonds(mol))
#        properties["stereo_double_bond_count"].append(count_stereo_double_bonds(mol))
        properties["heavy_atom_count"].append(count_heavy_atoms(mol))
#        properties["non_hydrogen_bond_count"].append(count_non_hydrogen_bonds(mol))
#        properties["positive_formal_charge_count"].append(count_positive_formal_charge_atoms(mol))
#        properties["negative_formal_charge_count"].append(count_negative_formal_charge_atoms(mol))
        properties["net_formal_charge"].append(get_net_formal_charge(mol))
        properties["sorted_rings"].append(sorted_rings(mol))
        properties["kekulized_smiles"].append(kekulized_smiles(mol))
        properties["kekulized_smiles_atom_map"].append(kekulized_smiles(mol, atom_map=True))
        properties["hybridization_indices"].append(get_hybridization_indices(mol))
        properties["element_counts"].append(get_element_counts(mol))
        properties["bond_counts"].append(get_bond_counts(mol))
        properties["ring_counts"].append(get_ring_counts(mol))
        properties["stereo_summary"].append(get_stereo_summary(mol))
        properties["molecular_formula"].append(get_molecular_formula(mol))

    return properties
