#!/usr/bin/env python3
"""Write back the cross-view mode winner into a bimodal run's dense depth maps.

GT-free: for each flagged pixel, the dominant hypothesis is the run's dense
depth and the alternate is dense * (z_alt/z_dom) from the canonical cache;
both are z-buffer-voted against the other cameras' dense maps (referees
masked at their own flagged pixels) and the alternate is written only if it
wins by > margin votes. Outputs a full run dir (depth_*.npy + result.npz)
usable as --run_b in compare_phase1.py.

Usage:
  python readout_writeback.py --scene ~/scratch/datasets/eth3d/kicker \
      --cache ~/scratch/mast3r_out/p1cache/kicker \
      --run ~/scratch/mast3r_out/bimodal_p2/kicker \
      --out ~/scratch/mast3r_out/bimodal_p2_rb/kicker --tau 0.15
"""
import argparse
import shutil
from pathlib import Path

import numpy as np

from readout_check import EPS, consistency_votes, load_canon2_pair


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True, help='bimodal run out dir')
    p.add_argument('--out', type=Path, required=True, help='corrected run out dir')
    p.add_argument('--tau', type=float, default=0.15)
    p.add_argument('--rel_tol', type=float, default=0.08)
    p.add_argument('--margin', type=int, default=2)
    args = p.parse_args()

    res = np.load(args.run / 'result.npz', allow_pickle=True)
    names, cam2w, intrinsics = list(res['names']), res['cam2w'], res['intrinsics']
    depths = {n: np.load(args.run / f'depth_{Path(n).stem}.npy') for n in names}
    w2c = [np.linalg.inv(c) for c in cam2w]
    imgdir = args.scene / 'images' / 'dslr_images_undistorted'

    ratios, ref_depths = {}, {}
    for n in names:
        c2, c2_alt = load_canon2_pair(args.cache, imgdir / n, args.tau)
        ratios[n] = None if c2 is None else c2_alt / np.clip(c2, EPS, None)
        d = depths[n].copy()
        if ratios[n] is not None:
            d[np.abs(ratios[n] - 1.) > 1e-4] = np.nan
        d[d <= 0] = np.nan
        ref_depths[n] = d

    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.run / 'result.npz', args.out / 'result.npz')

    tot_flag = tot_switch = 0
    for a, name in enumerate(names):
        dense = depths[name].copy()
        ratio = ratios[name]
        if ratio is None:
            print(f'{name}: no bimodal canon cache (+alt), copied unchanged')
            np.save(args.out / f'depth_{Path(name).stem}.npy', dense)
            continue
        sel = (np.abs(ratio - 1.) > 1e-4) & (dense > 0)
        if sel.any():
            vv, uu = np.nonzero(sel)
            uv = np.stack([uu, vv], 1).astype(np.float64)
            z_dom = dense[sel]
            z_alt = z_dom * ratio[sel]
            others = [(intrinsics[j], w2c[j], ref_depths[names[j]])
                      for j in range(len(names)) if j != a]
            v_dom = consistency_votes(uv, z_dom, intrinsics[a], cam2w[a], others, args.rel_tol)
            v_alt = consistency_votes(uv, z_alt, intrinsics[a], cam2w[a], others, args.rel_tol)
            pick_alt = v_alt > v_dom + args.margin
            dense[vv[pick_alt], uu[pick_alt]] = z_alt[pick_alt]
            tot_flag += int(sel.sum())
            tot_switch += int(pick_alt.sum())
            print(f'{name}: flagged {sel.sum():6d}  switched {pick_alt.sum():6d} '
                  f'({pick_alt.mean():.1%})')
        np.save(args.out / f'depth_{Path(name).stem}.npy', dense.astype(np.float32))

    print(f'\nTotal: {tot_switch}/{tot_flag} flagged px switched to alternate '
          f'({tot_switch / max(tot_flag, 1):.1%}); corrected run at {args.out}')


if __name__ == '__main__':
    main()
