#!/usr/bin/env python3
"""Aggregate the seed-variance runs: per-scene mean/std/min/max over seeds.

Usage: python3 aggregate_variance.py [out_root]   (default ~/scratch/mast3r_out/variance)
"""
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else
            Path.home() / 'scratch/mast3r_out/variance')

runs = {}   # scene -> list of (seed, metrics)
for mfile in sorted(root.glob('seed*/*/metrics.json')):
    m = json.load(open(mfile))
    seed = int(mfile.parts[-3].removeprefix('seed'))
    runs.setdefault(m['scene'], []).append((seed, m))

for key in ('RRA@5', 'RTA@5', 'mAA(30)'):
    print(f'\n===== {key} =====')
    print(f"{'scene':<12} {'mean':>7} {'std':>7} {'min':>7} {'max':>7}  per-seed")
    for scene, rows in sorted(runs.items()):
        rows = sorted(rows)
        v = np.array([m[key] for _, m in rows])
        per_seed = ' '.join(f'{x:.3f}' for x in v)
        print(f'{scene:<12} {v.mean():7.3f} {v.std():7.3f} {v.min():7.3f} {v.max():7.3f}  [{per_seed}]')
        if len(rows) < 5:
            print(f'{"":<12} !! only {len(rows)} seeds found')
