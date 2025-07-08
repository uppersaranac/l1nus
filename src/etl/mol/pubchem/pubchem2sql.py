"""
create a program that will read pubchem sdf files in the same directory layout as found on the pubchem ftp site. 
* For each record in the file
  * convert the molecule into a smiles string using rdkit. 
* insert the smiles string and all attributes in the pubchem sdf file into a postgres table
* use psycopg3
* to create database: create database pubchem;
"""

import os
import psycopg3
from rdkit import Chem
from pathlib import Path


field_map = {
"PUBCHEM_COMPOUND_CID": {'cid': 'INTEGER PRIMARY KEY'},
"PUBCHEM_CACTVS_COMPLEXITY": {'complexity': 'REAL'},
"PUBCHEM_CACTVS_HBOND_ACCEPTOR": {'hba': 'REAL'},
"PUBCHEM_CACTVS_HBOND_DONOR": { 'hbd': 'REAL'},
"PUBCHEM_CACTVS_ROTATABLE_BOND":  {'rotatable_bonds': 'INTEGER'},
"PUBCHEM_IUPAC_NAME": {'iupac_name': 'VARCHAR(2047)'},
"PUBCHEM_IUPAC_INCHI": { 'inchi_key': 'VARCHAR(2047)'},
"PUBCHEM_IUPAC_INCHIKEY":  {'inchi_key': 'VARCHAR(2047)'},
"PUBCHEM_XLOGP3":  {'logp': 'REAL'},
"PUBCHEM_EXACT_MASS":  {'exact_mass': 'REAL'},
"PUBCHEM_MOLECULAR_FORMULA":  {'formula': 'VARCHAR(2047)'},
"PUBCHEM_MOLECULAR_WEIGHT":  {'exact_mw': 'REAL'},
"PUBCHEM_SMILES":  {'isomeric_smiles': 'VARCHAR(2047)'},
"PUBCHEM_CACTVS_TPSA":  {'tspa': 'REAL'},
"PUBCHEM_MONOISOTOPIC_WEIGHT":  {'mono_mass': 'REAL'},
"PUBCHEM_TOTAL_CHARGE":  {"charge": 'INTEGER'},
"PUBCHEM_HEAVY_ATOM_COUNT":  {'num_atoms': 'INTEGER'},
"PUBCHEM_ATOM_DEF_STEREO_COUNT":  {'num_def_stereo': 'INTEGER'},
"PUBCHEM_ATOM_UDEF_STEREO_COUNT":  {'num_undef_stereo': 'INTEGER'},
"PUBCHEM_BOND_DEF_STEREO_COUNT":  {'num_def_double': 'INTEGER'},
"PUBCHEM_BOND_UDEF_STEREO_COUNT":  {'num_undef_double': 'INTEGER'},
"PUBCHEM_ISOTOPIC_ATOM_COUNT":  {'num_isotopic': 'INTEGER'},
"PUBCHEM_COMPONENT_COUNT":  {'fragments': 'INTEGER'},
}


def create_compound_table(conn):
    cursor = conn.cursor()

    create_table = "CREATE TABLE compounds ("
    for k,v in field_map.items():
        entry = next(iter(v))
        create_table += f"{entry[0]} {entry[1]},"

    create_table = create_table[:-1] + ");"
    cursor.execute(create_table)
    conn.commit()
    cursor.close()

def extract_mapped_values(props):
    result = {}
    for prop, value in props.items():
        if prop in field_map:
            column_name = list(field_map[prop].keys())[0]
            result[column_name] = value
    return result

def process_sdf_file(file_path, conn):
    """Process a single SDF file and insert records into PostgreSQL."""
    suppl = Chem.SDMolSupplier(str(file_path))
    cursor = conn.cursor()
    
    for mol in suppl:
        if mol is None:
            continue
            
        # Generate SMILES string
        # smiles = Chem.MolToSmiles(mol)
        
        # Get all properties from the molecule
        props = mol.GetPropsAsDict()

        results = extract_mapped_values(props)
        
        # Prepare column names and values
        columns = list(results.keys())
        values = list(results.values())
        
        # Create INSERT statement
        placeholders = ','.join(['%s'] * len(columns))
        column_names = ','.join(columns)
        sql = f"INSERT INTO pubchem_compounds ({column_names}) VALUES ({placeholders})"
        
        # Execute insert
        cursor.execute(sql, values)
    
    conn.commit()

def main():
    # Database connection parameters
    db_params = {
        'dbname': 'pubchem',
        'user': 'postgres',
        'password': 'kingfarm',
        'host': 'localhost'
    }
    
    # Connect to PostgreSQL
    conn = psycopg3.connect(**db_params)

    create_compound_table(conn)
    
    # Root directory containing PubChem SDF files
    pubchem_dir = Path('pubchem_data')
    
    # Process all SDF files recursively
    for sdf_file in pubchem_dir.rglob('*.sdf'):
        try:
            process_sdf_file(sdf_file, conn)
            print(f"Processed: {sdf_file}")
        except Exception as e:
            print(f"Error processing {sdf_file}: {e}")
    
    conn.close()

if __name__ == '__main__':
    main()



