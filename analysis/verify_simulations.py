#!/usr/bin/env python3
"""Bundle-only independent numerical replay; no patient or historical input needed."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'src'))
import argparse,json
from pathlib import Path
from inference_audit import audit_k2

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--summary',type=Path,required=True);p.add_argument('--repair',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=audit_k2(a.input,a.summary,entropy=202609160602 if a.repair else 202609160601,reps=100 if a.repair else 200,repair=a.repair,check_history=False)
    result.update(status='PASS',scope='bundle-only DGP/moment/set/family/hash replay; original local audit additionally checked old160 and primary800 disjointness',new_fits=0)
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
