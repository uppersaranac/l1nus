import argparse
import glob
import gzip
import os
import sys
from rdkit import Chem
import pyarrow as pa
import pyarrow.ipc as ipc
from pathlib import Path
from multiprocessing import Pool, cpu_count
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

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
    'PUBCHEM_OPENEYE_ISO_SMILES': 'smiles',  # PUBCHEM_SMILES not in ftp sdf files
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
            return value
    except Exception:
        return None

def parse_molecule(mol):
    data = {}
    for prop, col in props2columns.items():
        value = mol.GetProp(prop) if mol.HasProp(prop) else None
        arrow_type = cast_map.get(col)
        data[col] = cast_value(value, arrow_type)
    return data

def process_file_to_arrow(filepath, max_records, index):
    output_path = f"temp_output_{index}.arrow"
    open_func = gzip.open if filepath.endswith(".gz") else open
    try:
        with open_func(filepath, 'rb') as f:
            try:
                suppl = Chem.ForwardSDMolSupplier(f)
                with ipc.new_file(output_path, schema) as writer:
                    batch = []
                    count = 0
                    for mol in suppl:
                        if mol is None:
                            continue
                        batch.append(parse_molecule(mol))
                        count += 1
                        if len(batch) >= 1000:
                            table = pa.Table.from_pylist(batch, schema=schema)
                            writer.write(table)
                            batch.clear()
                        if max_records and count >= max_records:
                            break
                    if batch:
                        table = pa.Table.from_pylist(batch, schema=schema)
                        writer.write(table)
            except Exception as e:
                print(f"[Error] RDKit supplier error in file {filepath}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"[Error] Failed to open file {filepath}: {e}", file=sys.stderr)
        sys.exit(1)
    return output_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf_files", nargs="+", required=True, help="Input SDF files (supports globbing)")
    parser.add_argument("--max_records", type=int, default=0, help="Max number of records to parse (0 = unlimited)")
    parser.add_argument("--output", required=True, help="Output Arrow file")
    parser.add_argument("--n_cores", type=int, default=cpu_count(), help="Number of cores to use")
    args = parser.parse_args()

    sdf_files = [f for pattern in args.sdf_files for f in glob.glob(pattern)]
    max_records = args.max_records if args.max_records > 0 else None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Processing SDF files...", total=len(sdf_files))

        with Pool(processes=args.n_cores) as pool:
            results = []
            for i, filepath in enumerate(sdf_files):
                results.append(pool.apply_async(process_file_to_arrow, args=(filepath, max_records, i)))
            pool.close()

            output_files = []
            for r in results:
                output_files.append(r.get())
                progress.update(task, advance=1)
    with pa.OSFile(str(args.output), 'wb') as sink:
        with pa.RecordBatchFileWriter(sink, schema) as writer:
            for path in output_files:
                table = pa.ipc.RecordBatchFileReader(pa.OSFile(str(path), 'r')).read_all()
                writer.write_table(table)

    # huggingface currently only supports arrow streaming format, not the random access format
    # writer = None
    # for path in output_files:
    #     with pa.ipc.RecordBatchFileReader(pa.memory_map(path,'r')) as dataset:
    #         for i in range(dataset.num_record_batches):
    #             rb = dataset.get_batch(i)
    #             if writer is None:
    #                 writer = pa.RecordBatchStreamWriter(str(args.output), rb.schema)
    #             writer.write_batch(rb)

    # writer.close()

if __name__ == "__main__":
    main()
    print("Done!")