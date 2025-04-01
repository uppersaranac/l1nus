import argparse
import glob
import gzip
import psycopg2
from rdkit import Chem
from rdkit.Chem import SDMolSupplier

"""
CREATE TABLE compound (
    cid INTEGER PRIMARY KEY,
    complexity DOUBLE PRECISION,
    hbond_acceptor INTEGER,
    hbond_donor INTEGER,
    rotatable_bond INTEGER,
    tpsa DOUBLE PRECISION,
    xlogp3 DOUBLE PRECISION,
    monoisotopic_weight DOUBLE PRECISION,
    exact_mass DOUBLE PRECISION,
    molecular_formula VARCHAR(1024),
    molecular_weight DOUBLE PRECISION,
    total_charge INTEGER,
    heavy_atom_count INTEGER,
    atom_def_stereo_count INTEGER,
    atom_udef_stereo_count INTEGER,
    bond_def_stereo_count INTEGER,
    bond_udef_stereo_count INTEGER,
    isotopic_atom_count INTEGER,
    component_count INTEGER,
    tauto_count INTEGER,
    complexity_atom_count INTEGER,
    iupac_openeye_name VARCHAR(1024),
    iupac_cas_name VARCHAR(1024),
    iupac_name_markup VARCHAR(1024),
    iupac_name VARCHAR(1024),
    iupac_systematic_name VARCHAR(1024),
    iupac_traditional_name VARCHAR(1024),
    smiles VARCHAR(1024),
    openeye_can_smiles VARCHAR(1024),
    openeye_iso_smiles VARCHAR(1024)
);
"""

# Map SDF property names to database column names
props2columns = {
    'PUBCHEM_COMPOUND_CID': 'cid',
    'PUBCHEM_CACTVS_COMPLEXITY': 'complexity',
    'PUBCHEM_CACTVS_HBOND_ACCEPTOR': 'hbond_acceptor',
    'PUBCHEM_CACTVS_HBOND_DONOR': 'hbond_donor',
    'PUBCHEM_CACTVS_ROTATABLE_BOND': 'rotatable_bond',
    'PUBCHEM_CACTVS_TPSA': 'tpsa',
    'PUBCHEM_XLOGP3': 'xlogp3',
    'PUBCHEM_MONOISOTOPIC_WEIGHT': 'monoisotopic_weight',
    'PUBCHEM_EXACT_MASS': 'exact_mass',
    'PUBCHEM_MOLECULAR_FORMULA': 'molecular_formula',
    'PUBCHEM_MOLECULAR_WEIGHT': 'molecular_weight',
    'PUBCHEM_TOTAL_CHARGE': 'total_charge',
    'PUBCHEM_HEAVY_ATOM_COUNT': 'heavy_atom_count',
    'PUBCHEM_ATOM_DEF_STEREO_COUNT': 'atom_def_stereo_count',
    'PUBCHEM_ATOM_UDEF_STEREO_COUNT': 'atom_udef_stereo_count',
    'PUBCHEM_BOND_DEF_STEREO_COUNT': 'bond_def_stereo_count',
    'PUBCHEM_BOND_UDEF_STEREO_COUNT': 'bond_udef_stereo_count',
    'PUBCHEM_ISOTOPIC_ATOM_COUNT': 'isotopic_atom_count',
    'PUBCHEM_COMPONENT_COUNT': 'component_count',
    'PUBCHEM_CACTVS_TAUTO_COUNT': 'tauto_count',
    'PUBCHEM_CACTVS_COMPLEXITY_ATOM_COUNT': 'complexity_atom_count',
    'PUBCHEM_IUPAC_OPENEYE_NAME': 'iupac_openeye_name',
    'PUBCHEM_IUPAC_CAS_NAME': 'iupac_cas_name',
    'PUBCHEM_IUPAC_NAME_MARKUP': 'iupac_name_markup',
    'PUBCHEM_IUPAC_NAME': 'iupac_name',
    'PUBCHEM_IUPAC_SYSTEMATIC_NAME': 'iupac_systematic_name',
    'PUBCHEM_IUPAC_TRADITIONAL_NAME': 'iupac_traditional_name',
    'PUBCHEM_SMILES': 'smiles',
    'PUBCHEM_OPENEYE_CAN_SMILES': 'openeye_can_smiles',
    'PUBCHEM_OPENEYE_ISO_SMILES': 'openeye_iso_smiles'
}

# Value casting rules
double_fields = {
    'complexity', 'tpsa', 'xlogp3', 'monoisotopic_weight', 'exact_mass', 'molecular_weight'
}
int_fields = {
    'cid', 'hbond_acceptor', 'hbond_donor', 'rotatable_bond', 'total_charge',
    'heavy_atom_count', 'atom_def_stereo_count', 'atom_udef_stereo_count',
    'bond_def_stereo_count', 'bond_udef_stereo_count', 'isotopic_atom_count',
    'component_count', 'tauto_count', 'complexity_atom_count'
}
text_fields = set(props2columns.values()) - double_fields - int_fields

def cast_value(field, value):
    if value is None or value == '':
        return None
    try:
        if field in double_fields:
            return float(value)
        elif field in int_fields:
            return int(value)
        return str(value)
    except:
        return None

def parse_sdf_files(files, limit):
    data = []
    count = 0
    for file in files:
        print(f"Reading: {file}")
        if file.endswith('.gz'):
            f = gzip.open(file, 'rb')
            suppl = Chem.ForwardSDMolSupplier(f)
        else:
            suppl = Chem.SDMolSupplier(file)
       
        for mol in suppl:
            if mol is None:
                continue
            row = {}
            for sdf_prop, db_col in props2columns.items():
                val = mol.GetProp(sdf_prop) if mol.HasProp(sdf_prop) else None
                row[db_col] = cast_value(db_col, val)
            data.append(row)
            count += 1
            if limit and count >= limit:
                return data
        if file.endswith('.gz'):
            f.close()
    return data

def insert_into_postgresql(data, db_params):
    if not data:
        print("No data to insert.")
        return

    columns = list(data[0].keys())
    placeholders = ', '.join(['%s'] * len(columns))
    colnames = ', '.join(columns)
    sql = f"INSERT INTO public.compound ({colnames}) VALUES ({placeholders}) ON CONFLICT (cid) DO NOTHING"

    conn = psycopg2.connect(**db_params)
    with conn:
        with conn.cursor() as cur:
            for row in data:
                cur.execute(sql, [row[col] for col in columns])
    print(f"Inserted {len(data)} rows into compound.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sdf', required=True, help="Glob pattern for SDF files (e.g. 'data/*.sdf')")
    parser.add_argument('--limit', type=int, default=None, help="Limit the number of records to load")
    parser.add_argument('--dbhost', default='localhost')
    parser.add_argument('--dbname', default='chem')
    parser.add_argument('--dbuser', default='postgres')
    parser.add_argument('--dbpass', default='kingfarm')
    parser.add_argument('--dbport', default=5432, type=int)

    args = parser.parse_args()
    files = glob.glob(args.sdf)
    if not files:
        print("No files matched.")
        return

    data = parse_sdf_files(files, args.limit)
    db_params = {
        'host': args.dbhost,
        'database': args.dbname,
        'user': args.dbuser,
        'password': args.dbpass,
        'port': args.dbport
    }
    insert_into_postgresql(data, db_params)

if __name__ == '__main__':
    main()