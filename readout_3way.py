#!/usr/bin/env python3
"""3-way mode readout on a BASELINE (avg-angle) run: {avg, dom, alt} hypotheses.

Unlike readout_writeback.py (which corrects a bimodal run and inherits its
coin-flip dominant-mode choice on undecided pixels), this keeps the baseline
average everywhere and replaces it only where a mode hypothesis decisively
out-votes it in cross-view consistency. Fallback = the average -> structurally
no-worse-than-baseline up to vote accuracy.

Hypotheses at flagged px: z0 = baseline dense; z_dom/z_alt = z0 scaled by the
canonical mode/avg depth ratios from the caches (avg-angle + bimodal + .alt).

Usage:
  python readout_3way.py --scene ~/scratch/datasets/eth3d/kicker \
      --cache ~/scratch/mast3r_out/p1cache/kicker \
      --run ~/scratch/mast3r_out/baseline_s0b/kicker \
      --out ~/scratch/mast3r_out/baseline_s0b_3way/kicker --tau 0.15
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

from mast3r.utils.misc import hash_md5
from readout_check import EPS, consistency_votes


def load_canon2(cache_dir, img_path, mode, subsample=8):
    kw = {'mode': mode}
    f = cache_dir / 'canon_views' / (hash_md5(str(img_path)) + f'_{subsample=}_{kw=}.pth')
    if not f.exists():
        return None
    (canon, canon2, cconf), focal = torch.load(f, map_location='cpu', weights_only=False)
    return canon2.numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True,
                   help='canon_views dir with avg-angle AND bimodal(+.alt) caches')
    p.add_argument('--run', type=Path, required=True, help='BASELINE run out dir')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tau', type=float, default=0.15)
    p.add_argument('--rel_tol', type=float, default=0.08)
    p.add_argument('--margin', type=int, default=2,
                   help='a mode must beat the average by > margin votes')
    args = p.parse_args()

    res = np.load(args.run / 'result.npz', allow_pickle=True)
    names, cam2w, intrinsics = list(res['names']), res['cam2w'], res['intrinsics']
    depths = {n: np.load(args.run / f'depth_{Path(n).stem}.npy') for n in names}
    w2c = [np.linalg.inv(c) for c in cam2w]
    imgdir = args.scene / 'images' / 'dslr_images_undistorted'

    r_dom, r_alt, ref_depths = {}, {}, {}
    for n in names:
        c2_avg = load_canon2(args.cache, imgdir / n, 'avg-angle')
        c2_dom = load_canon2(args.cache, imgdir / n, f'bimodal-{args.tau}')
        altf = args.cache / 'canon_views' / (
            hash_md5(str(imgdir / n)) + f"_subsample=8_kw={{'mode': 'bimodal-{args.tau}'}}.pth.alt")
        d = depths[n].copy()
        if c2_avg is None or c2_dom is None or not altf.exists():
            r_dom[n] = r_alt[n] = None
        else:
            c2_alt = torch.load(str(altf), map_location='cpu', weights_only=False).numpy()
            r_dom[n] = c2_dom / np.clip(c2_avg, EPS, None)
            r_alt[n] = c2_alt / np.clip(c2_avg, EPS, None)
            d[np.abs(c2_alt / np.clip(c2_dom, EPS, None) - 1.) > 1e-4] = np.nan
        d[d <= 0] = np.nan
        ref_depths[n] = d

    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.run / 'result.npz', args.out / 'result.npz')

    tot_flag = tot_switch = 0
    for a, name in enumerate(names):
        dense = depths[name].copy()
        if r_dom[name] is None:
            print(f'{name}: missing canon caches, copied unchanged')
            np.save(args.out / f'depth_{Path(name).stem}.npy', dense)
            continue
        flag = np.abs(r_alt[name] / np.clip(r_dom[name], EPS, None) - 1.) > 1e-4
        sel = flag & (dense > 0)
        if sel.any():
            vv, uu = np.nonzero(sel)
            uv = np.stack([uu, vv], 1).astype(np.float64)
            z0 = dense[sel]
            zd = z0 * r_dom[name][sel]
            za = z0 * r_alt[name][sel]
            others = [(intrinsics[j], w2c[j], ref_depths[names[j]])
                      for j in range(len(names)) if j != a]
            v0 = consistency_votes(uv, z0, intrinsics[a], cam2w[a], others, args.rel_tol)
            vd = consistency_votes(uv, zd, intrinsics[a], cam2w[a], others, args.rel_tol)
            va = consistency_votes(uv, za, intrinsics[a], cam2w[a], others, args.rel_tol)
            z_mode = np.where(vd >= va, zd, za)
            v_mode = np.maximum(vd, va)
            pick = v_mode > v0 + args.margin
            dense[vv[pick], uu[pick]] = z_mode[pick]
            tot_flag += int(sel.sum())
            tot_switch += int(pick.sum())
            print(f'{name}: flagged {sel.sum():6d}  switched {pick.sum():6d} '
                  f'({pick.mean():.1%})  [dom {(pick & (vd >= va)).sum()} / '
                  f'alt {(pick & (vd < va)).sum()}]')
        np.save(args.out / f'depth_{Path(name).stem}.npy', dense.astype(np.float32))

    print(f'\nTotal: {tot_switch}/{tot_flag} flagged px replaced by a mode '
          f'({tot_switch / max(tot_flag, 1):.1%}); corrected run at {args.out}')


if __name__ == '__main__':
    main()
