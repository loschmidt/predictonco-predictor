#!/usr/bin/env python3

import argparse
import csv
import json
import sys
from typing import List

from predictor import OutputDict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert PredictONCO Predictor output from JSON to TSV')
    parser.add_argument('-i', '--input', type=lambda f: open(f, 'r'), default=sys.stdin)
    parser.add_argument('-o', '--output', type=lambda f: open(f, 'w'), default=sys.stdout)
    parser.add_argument('-d', '--delimiter', default='\t')
    args = parser.parse_args()

    data: List[OutputDict] = json.load(args.input)

    fieldnames = ['id', 'decision', 'confidence']
    writer = csv.DictWriter(args.output, fieldnames=fieldnames, delimiter=args.delimiter)
    writer.writeheader()
    writer.writerows(data)
