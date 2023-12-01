#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from typing import Literal, TypedDict, List, Union

import numpy as np
import xgboost as xgb


# sequential features
class SeqInputDict(TypedDict):
    id: str
    structure: bool
    protein_type: Literal['PROTO_ONCOGENE', 'TUMOR_SUPPRESSOR']
    essential: bool
    domain: Literal['cytoplasmic', 'extracellular', 'transmembrane', 'other']
    predictsnp: float
    essential_residues_all: int
    conservation: int
    msa_data: float


# structural features (extend sequential features)
class InputDict(SeqInputDict):
    pocket: bool
    foldx: float
    rosetta: float
    pka_num: int
    pka_min: float
    pka_max: float


class OutputDict(TypedDict):
    id: str
    decision: Literal['DELETERIOUS', 'BENIGN']
    confidence: float


def predict(data: Union[SeqInputDict, InputDict], xgb_seq: xgb.XGBClassifier, xgb_struc: xgb.XGBClassifier,
            col_seq: List[str], col_struc: List[str]) -> OutputDict:
    
    if data['structure']:
        x = [data[name] for name in col_struc]
        x[6] = (x[6] == 'PROTO_ONCOGENE')
        x = np.array([x])
        confidence = 100 * xgb_struc.predict_proba(x)[:,1]
    else:
        x = [data[name] for i, name in enumerate(col_seq) if i < 6] 
        x_dom = [data['domain'] == name for i, name in enumerate(col_seq) if i >= 6]
        x[0] = (x[0] == 'PROTO_ONCOGENE')
        x = np.array([x + x_dom])
        confidence = 100 * xgb_seq.predict_proba(x)[:,1]

    confidence = float(confidence[0])
    
    cutoff_decision = 50

    if confidence >= cutoff_decision:
        decision = 'DELETERIOUS'
    else:
        decision = 'BENIGN'
        confidence = 100 - confidence

    return OutputDict(
        id=data['id'],
        decision=decision,
        confidence=confidence,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PredictONCO Predictor')
    parser.add_argument('-i', '--input', type=lambda f: open(f, 'r'), default=sys.stdin)
    parser.add_argument('-o', '--output', type=lambda f: open(f, 'w'), default=sys.stdout)
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    
    content = json.load(args.input)
    
    xgb_seq = xgb.XGBClassifier()
    xgb_seq.load_model(script_dir / 'xgb_seq.json')
    col_seq = []
    with open(script_dir / 'cols_seq.txt') as f:
        for line in f:
            col_seq.append(line.strip())
    
    xgb_struc = xgb.XGBClassifier()
    xgb_struc.load_model(script_dir / 'xgb_struc.json')
    col_struc = []
    with open(script_dir / 'cols_struc.txt') as f:
        for line in f:
            col_struc.append(line.strip())

    if type(content) is list:
        # content: List[InputDict]
        result = [predict(i, xgb_seq, xgb_struc, col_seq, col_struc) for i in content]
    else:
        # content: InputDict
        result = predict(content, xgb_seq, xgb_struc, col_seq, col_struc)

    json.dump(result, args.output)
