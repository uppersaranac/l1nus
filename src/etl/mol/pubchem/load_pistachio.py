import argparse
import os
import glob

import pyarrow as pa
from rdkit import Chem
from rdkit.Chem import AllChem

# Utility to canonicalize a list of SMILES using RDKit
def canonicalize_smiles_list(smiles_list):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list if smi]
    cano = [Chem.MolToSmiles(mol, canonical=True) for mol in mols if mol]
    return sorted(set(cano))


def parse_args():
    parser = argparse.ArgumentParser(description='Deduplicate reaction SMILES and write to Arrow file.')
    parser.add_argument('--input', type=str, required=True, help='Glob pattern for input files containing reaction SMILES.')
    parser.add_argument('--output', type=str, required=True, help='Output Arrow file name.')
    parser.add_argument('--limit', type=int, default=None, help='Maximum number of reactions to process.')
    parser.add_argument('--skip-header', action='store_true', default=True, help='Skip the first line of each input file (default: True).')
    return parser.parse_args()

def parse_reaction_smiles_rdkit(line):
    """
    Parse a reaction SMILES line using RDKit, extracting canonicalized reactants, agents, and products from the parsed reaction object.
    """
    rxn_smiles = line.strip()
    try:
        rxn = AllChem.ReactionFromSmarts(rxn_smiles, useSmiles=True)
    except Exception as e:
        raise ValueError(f"RDKit could not parse: {rxn_smiles} ({e})")
    if rxn is None:
        raise ValueError(f"RDKit returned None for: {rxn_smiles}")
    # Extract reactants, agents, products as canonical SMILES
    def extract_smiles_templates(get_template, num_templates):
        smiles = []
        for i in range(num_templates):
            mol = get_template(i)
            if mol:
                for atom in mol.GetAtoms():
                    atom.SetAtomMapNum(0)
                smi = Chem.MolToSmiles(mol, canonical=True)
                if smi:
                    smiles.append(smi)
        return tuple(sorted(set(smiles)))

    reactants = extract_smiles_templates(rxn.GetReactantTemplate, rxn.GetNumReactantTemplates())
    agents = extract_smiles_templates(rxn.GetAgentTemplate, rxn.GetNumAgentTemplates())
    products = extract_smiles_templates(rxn.GetProductTemplate, rxn.GetNumProductTemplates())
    return (reactants, agents, products)


def main():
    args = parse_args()
    input_pattern = os.path.expanduser(args.input)
    files = glob.glob(input_pattern)

    seen = set()
    reactant_list = []
    agent_list = []
    product_list = []
    count = 0

    for fname in files:
        with open(fname, 'r') as f:
            if args.skip_header:
                next(f, None)
            for line in f:
                if not line.strip():
                    continue
                try:
                    rxn = parse_reaction_smiles_rdkit(line)
                except Exception as e:
                    print(f"Skipping invalid line in {fname}: {line.strip()} ({e})")
                    continue
                # Deduplication by canonical tuple
                key = rxn
                if key in seen:
                    continue
                seen.add(key)
                reactant_list.append(list(rxn[0]))
                agent_list.append(list(rxn[1]))
                product_list.append(list(rxn[2]))
                count += 1
                if args.limit and count >= args.limit:
                    break
        if args.limit and count >= args.limit:
            break

    # Write to Arrow file
    table = pa.table({
        'reactants': reactant_list,
        'agents': agent_list,
        'products': product_list
    })
    import pyarrow.ipc as ipc
    with pa.OSFile(args.output, 'wb') as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    print(f"Wrote {len(reactant_list)} deduplicated reactions to {args.output}")

if __name__ == '__main__':
    main()
