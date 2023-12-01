#!/usr/bin/env python3

import argparse
import csv
import json
import sys
from typing import List, Union

from predictor import InputDict, SeqInputDict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert PredictONCO Predictor input from TSV to JSON')
    parser.add_argument('-i', '--input', type=lambda f: open(f, 'r'), default=sys.stdin)
    parser.add_argument('-o', '--output', type=lambda f: open(f, 'w'), default=sys.stdout)
    parser.add_argument('-d', '--delimiter', default='\t')
    args = parser.parse_args()

    reader = csv.DictReader(args.input, delimiter=args.delimiter)
    
    if reader.fieldnames is None:
        raise Exception("reader.fieldnames is None")
    
    data: List[Union[InputDict, SeqInputDict]] = []
    for row in reader:
        protein_type = row['protein_type']
        domain = row['domain']
        if protein_type not in ('PROTO_ONCOGENE', 'TUMOR_SUPPRESSOR'):
            raise Exception('Protein type is ' + protein_type)
        if domain not in ('cytoplasmic', 'extracellular', 'transmembrane', 'other'):
            raise Exception('Domain is ' + domain)
        structural = bool(int(row['structure']))
        d = SeqInputDict(
            id=row['id'],
            structure=structural,
            protein_type=protein_type,  # type: ignore
            essential=row['essential'] == "1",
            domain=domain,  # type: ignore
            predictsnp=float(row['predictsnp']),
            essential_residues_all=int(row['essential_residues_all']),
            conservation=int(row['conservation']),
            msa_data=float(row['msa_data']),
        )
        if structural:
            d = InputDict(
                **d,
                pocket=row['pocket'] == "1",
                foldx=float(row['foldx']),
                rosetta=float(row['rosetta']),
                pka_num=int(row['pka_num']),
                pka_min=float(row['pka_min'] or "0"),
                pka_max=float(row['pka_max'] or "0"),
            )
        data.append(d)

    json.dump(data, args.output)
