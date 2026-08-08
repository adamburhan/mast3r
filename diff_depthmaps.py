#!/usr/bin/env python3
"""Are two runs' dense depth maps actually different, and where?

Compares depth_*.npy between two run dirs (same scene, same seed). Reports the
per-image median |log ratio| after global scale alignment (independent runs
differ in gauge), stratified by the bimodal flag mask if --cache/--scene given.
If maps are bit-identical the mixture never engaged -> implementation bug.

Usage:
  python diff_depthmaps.py --run_a .../p2_base/relief_s0 --run_b .../p2_mix/relief_s0 \
      --scene ~/scratch/datasets/eth3d/relief --cache .../p2cache/relief_s0 --tau 0.15
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from mast3r.utils.misc import hash_md5

EPS = 1e-8


def load_flag(cache_dir, img_path, tau, subsample=8):
    kw = {'mode': f'bimodal-{tau}'}
    f = cache_dir / 'canon_views' / (hash_md5(str(img_path)) + f'_{subsample=}_{kw=}.pth')
    if not f.exists() or not Path(str(f) + '.alt').exists():
        return None
    (canon, canon2, cconf), focal = torch.load(f, map_location='cpu', weights_only=False)
    canon2_alt = torch.load(str(f) + '.alt', map_location='cpu', weights_only=False)
    return np.abs(canon2_alt.numpy() / np.clip(canon2.numpy(), EPS, None) - 1.) > 1e-4


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run_a', type=Path, required=True)
    p.add_argument('--run_b', type=Path, required=True)
    p.add_argument('--scene', type=Path, default=None, help='for flag stratification')
    p.add_argument('--cache', type=Path, default=None, help='bimodal canon cache dir')
    p.add_argument('--tau', type=float, default=0.15)
    args = p.parse_args()
    imgdir = args.scene / 'images' / 'dslr_images_undistorted' if args.scene else None

    files = sorted(f.name for f in args.run_a.glob('depth_*.npy')
                   if (args.run_b / f.name).exists())
    if not files:
        print('no common depth files'); return

    acc = {k: [] for k in ('med', 'p99', 'med_f', 'med_u')}
    n_ident = 0
    for fn in files:
        da, db = np.load(args.run_a / fn), np.load(args.run_b / fn)
        if np.array_equal(da, db):
            n_ident += 1
            print(f'{fn}: IDENTICAL')
            continue
        ok = (da > 0) & (db > 0)
        dlog = np.log(db[ok]) - np.log(da[ok])
        d = np.full(da.shape, np.nan, np.float32)
        d[ok] = np.abs(dlog - np.median(dlog))   # remove global gauge shift

        row = dict(med=np.nanmedian(d), p99=np.nanpercentile(d, 99))
        msg = f"{fn}: med {row['med']:.4f}  p99 {row['p99']:.4f}"
        if imgdir and args.cache:
            stem = fn[len('depth_'):-len('.npy')]
            flag = load_flag(args.cache, imgdir / f'{stem}.JPG', args.tau)
            if flag is not None:
                row['med_f'] = np.nanmedian(d[flag]) if flag.any() else np.nan
                row['med_u'] = np.nanmedian(d[~flag])
                msg += f"  |  flagged {row['med_f']:.4f}  unflagged {row['med_u']:.4f}"
        for k, v in row.items():
            acc[k].append(v)
        print(msg)

    print(f'\n===== {len(files)} images, {n_ident} identical =====')
    for k, label in (('med', 'median |dlog| (gauge-aligned)'), ('p99', 'p99'),
                     ('med_f', 'median @flagged'), ('med_u', 'median @unflagged')):
        if acc[k]:
            print(f'  {label:32s} {np.nanmean(acc[k]):.4f}')


if __name__ == '__main__':
    main()
