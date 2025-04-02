import argparse
import glob
import gzip
import os
from rdkit import Chem
import pyarrow as pa
import pyarrow.ipc as ipc
from pathlib import Path

# props2columns dictionary from earlier
props2columns = {
    'PUBCHEM_COMPOUND_CID': 'cid',
    'PUBCHEM_CACTVS_COMPLEXITY': 'complexity',
    'PUBCHEM_CACTVS_HBOND_ACCEPTOR': 'hba',
    'PUBCHEM_CACTVS_HBOND_DONOR': 'hbd',
    'PUBCHEM_CACTVS_ROTATABLE_BOND': 'rotatable_bonds',
    'PUBCHEM_CACTVS_TPSA': 'tpsa',
    'PUBCHEM_XLOGP3': 'logp',
    'PUBCHEM_MONOISOTOPIC_WEIGHT': 'monoisotopic_mass',
    'PUBCHEM_EXACT_MASS': 'exact_mass',
    'PUBCHEM_MOLECULAR_FORMULA': 'formula',
    'PUBCHEM_MOLECULAR_WEIGHT': 'molecular_weight',
    'PUBCHEM_TOTAL_CHARGE': 'charge',
    'PUBCHEM_HEAVY_ATOM_COUNT': 'num_atoms',
    'PUBCHEM_ATOM_DEF_STEREO_COUNT': 'num_def_stereo',
    'PUBCHEM_ATOM_UDEF_STEREO_COUNT': 'num_undef_stereo',
    'PUBCHEM_BOND_DEF_STEREO_COUNT': 'num_def_double',
    'PUBCHEM_BOND_UDEF_STEREO_COUNT': 'num_undef_double',
    'PUBCHEM_ISOTOPIC_ATOM_COUNT': 'num_isotopic_atoms',
    'PUBCHEM_COMPONENT_COUNT': 'fragments',
    'PUBCHEM_CACTVS_TAUTO_COUNT': 'num_tautomers',
    'PUBCHEM_CACTVS_COMPLEXITY_ATOM_COUNT': 'num_complexity',
    'PUBCHEM_IUPAC_OPENEYE_NAME': 'iupac_openeye',
    'PUBCHEM_IUPAC_CAS_NAME': 'iupac_cas',
    'PUBCHEM_IUPAC_NAME': 'iupac',
    'PUBCHEM_IUPAC_SYSTEMATIC_NAME': 'iupac_systematic',
    'PUBCHEM_IUPAC_TRADITIONAL_NAME': 'iupac_traditional',
    'PUBCHEM_OPENEYE_ISO_SMILES': 'smiles',
}

# Define schema with proper types for Arrow output, derived from provided schema info
property_schema_fields = [
    pa.field("cid", pa.uint64()),
    pa.field("complexity", pa.float32()),
    pa.field("hba", pa.int32()),
    pa.field("hbd", pa.int32()),
    pa.field("rotatable_bonds", pa.int32()),
    pa.field("tpsa", pa.float32()),
    pa.field("logp", pa.float32()),
    pa.field("monoisotopic_mass", pa.float64()),
    pa.field("exact_mass", pa.float64()),
    pa.field("formula", pa.string()),
    pa.field("molecular_weight", pa.float64()),
    pa.field("charge", pa.int32()),
    pa.field("num_atoms", pa.int32()),
    pa.field("num_def_stereo", pa.int32()),
    pa.field("num_undef_stereo", pa.int32()),
    pa.field("num_def_double", pa.int32()),
    pa.field("num_undef_double", pa.int32()),
    pa.field("num_isotopic_atoms", pa.int32()),
    pa.field("fragments", pa.int32()),
    pa.field("num_tautomers", pa.int32()),
    pa.field("num_complexity", pa.int32()),
    pa.field("iupac_openeye", pa.string()),
    pa.field("iupac_cas", pa.string()),
    pa.field("iupac", pa.string()),
    pa.field("iupac_systematic", pa.string()),
    pa.field("iupac_traditional", pa.string()),
    pa.field("smiles", pa.string()),
]

schema = pa.schema(property_schema_fields)

# Build type casting map from schema
cast_map = {field.name: field.type for field in schema}

def cast_value(value, arrow_type):
    if value is None:
        return None
    try:
        if pa.types.is_integer(arrow_type):
            return int(value)
        elif pa.types.is_floating(arrow_type):
            return float(value)
        elif pa.types.is_boolean(arrow_type):
            return value.lower() in ("true", "1")
        elif pa.types.is_string(arrow_type):
            return str(value)
        else:
            return value  # default fallback
    except Exception:
        return None

def parse_molecule(mol):
    data = {}
    for prop, col in props2columns.items():
        value = mol.GetProp(prop) if mol.HasProp(prop) else None
        arrow_type = cast_map.get(col)
        data[col] = cast_value(value, arrow_type)
    return data


def stream_sdf_files(filepaths, max_records):
    count = 0
    for filepath in filepaths:
        open_func = gzip.open if filepath.endswith(".gz") else open
        with open_func(filepath, 'rb') as f:
            suppl = Chem.ForwardSDMolSupplier(f)
            for mol in suppl:
                if mol is None:
                    continue
                # prop_names = mol.GetPropNames()
                # for name in prop_names:
                #     value = mol.GetProp(name)
                #     print(f'{name}: {value}')
                yield parse_molecule(mol)
                count += 1
                if max_records and count >= max_records:
                    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf_files", nargs="+", required=True, help="Input SDF files (supports globbing)")
    parser.add_argument("--max_records", type=int, default=0, help="Max number of records to parse (0 = unlimited)")
    parser.add_argument("--output", required=True, help="Output Arrow file")
    args = parser.parse_args()

    sdf_files = [f for pattern in args.sdf_files for f in glob.glob(pattern)]
    max_records = args.max_records if args.max_records > 0 else None

    writer = None
    batch_size = 1000
    batch = []

    with ipc.new_file(args.output, schema) as writer:
        for record in stream_sdf_files(sdf_files, max_records):
            batch.append(record)
            if len(batch) >= batch_size:
                table = pa.Table.from_pylist(batch, schema=schema)
                writer.write(table)
                batch.clear()

        if batch:
            table = pa.Table.from_pylist(batch, schema=schema)
            writer.write(table)


if __name__ == "__main__":
    main()