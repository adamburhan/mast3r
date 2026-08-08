#!/usr/bin/env python3
"""Prototype + validate the mode-winner readout as post-processing (no GA changes).

For each flagged pixel of a bimodal run: the run's dense depth IS the dominant
hypothesis (optimized scale); the alternate is dense * (z_alt/z_dom) from the
cached canonical maps. Each hypothesis is backprojected with the optimized
poses and checked for depth-consistency against the other cameras' dense maps
(z-buffer vote). Reports: readout accuracy vs the GT-preferred mode, and the
dense flagged error if winners were written back, vs dominant and GT-oracle.

Usage:
  python readout_check.py --scene ~/scratch/datasets/eth3d/kicker \
      --cache ~/scratch/mast3r_out/p1cache/kicker \
      --run ~/scratch/mast3r_out/bimodal_p2/kicker --tau 0.15
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mast3r.utils.misc import hash_md5

EPS = 1e-8


def load_canon2_pair(cache_dir, img_path, tau, subsample=8):
    kw = {'mode': f'bimodal-{tau}'}
    f = cache_dir / 'canon_views' / (hash_md5(str(img_path)) + f'_{subsample=}_{kw=}.pth')
    if not f.exists() or not Path(str(f) + '.alt').exists():
        return None, None
    (canon, canon2, cconf), focal = torch.load(f, map_location='cpu', weights_only=False)
    canon2_alt = torch.load(str(f) + '.alt', map_location='cpu', weights_only=False)
    return canon2.numpy(), canon2_alt.numpy()


def load_gt(gt_dir, stem, net_hw):
    f = gt_dir / f'{stem}.npy'
    if not f.exists():
        return None
    gt = np.load(f).astype(np.float32)
    H0, W0 = gt.shape
    H1, W1 = net_hw
    s = 512. / max(W0, H0)
    Wi, Hi = round(W0 * s), round(H0 * s)
    gt = np.asarray(Image.fromarray(gt, mode='F').resize((Wi, Hi), Image.NEAREST))
    return gt[(Hi - H1) // 2:(Hi - H1) // 2 + H1, (Wi - W1) // 2:(Wi - W1) // 2 + W1]


def consistency_votes(uv, z_hyp, K_a, c2w_a, others, rel_tol=0.08):
    """uv: (P,2) pixel coords in image a; z_hyp: (P,) depths.
    others: list of (K_j, w2c_j, depth_j). Returns (P,) vote counts."""
    P = len(z_hyp)
    xyz_cam = np.concatenate([uv + 0.5, np.ones((P, 1))], 1) @ np.linalg.inv(K_a).T
    xyz_w = (xyz_cam * z_hyp[:, None]) @ c2w_a[:3, :3].T + c2w_a[:3, 3]
    votes = np.zeros(P)
    for K_j, w2c_j, depth_j in others:
        Hj, Wj = depth_j.shape
        xj = xyz_w @ w2c_j[:3, :3].T + w2c_j[:3, 3]
        zj = xj[:, 2]
        ok = zj > 0.05
        uvj = (xj @ K_j.T)
        uvj = uvj[:, :2] / np.clip(uvj[:, 2:], EPS, None)
        ui, vi = np.round(uvj[:, 0] - 0.5).astype(int), np.round(uvj[:, 1] - 0.5).astype(int)
        ok &= (ui >= 0) & (ui < Wj) & (vi >= 0) & (vi < Hj)
        dj = np.where(ok, depth_j[np.clip(vi, 0, Hj - 1), np.clip(ui, 0, Wj - 1)], np.nan)
        votes += (np.abs(zj - dj) / np.clip(zj, EPS, None) < rel_tol) & ok
    return votes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True, help='bimodal run out dir')
    p.add_argument('--gt_dir', type=Path, default=None)
    p.add_argument('--tau', type=float, default=0.15)
    p.add_argument('--rel_tol', type=float, default=0.08)
    p.add_argument('--margin', type=int, default=2,
                   help='override dominant only if alt wins by > margin votes')
    args = p.parse_args()
    gt_dir = args.gt_dir or args.scene / 'gt_depth_render'

    res = np.load(args.run / 'result.npz', allow_pickle=True)
    names, cam2w, intrinsics = list(res['names']), res['cam2w'], res['intrinsics']
    depths = {n: np.load(args.run / f'depth_{Path(n).stem}.npy') for n in names}
    w2c = [np.linalg.inv(c) for c in cam2w]
    imgdir = args.scene / 'images' / 'dslr_images_undistorted'

    # referee depths: mask out each referee's own flagged pixels so correlated
    # wrong-mode regions cannot vote (their depth there is unreliable)
    ref_depths = {}
    for n in names:
        c2, c2_alt = load_canon2_pair(args.cache, imgdir / n, args.tau)
        d = depths[n].copy()
        if c2 is not None:
            d[np.abs(c2_alt / np.clip(c2, EPS, None) - 1.) > 1e-4] = np.nan
        d[d <= 0] = np.nan
        ref_depths[n] = d

    acc = {k: [] for k in ('acc', 'err_dom', 'err_read', 'err_orc', 'nflag')}
    for a, name in enumerate(names):
        c2, c2_alt = load_canon2_pair(args.cache, imgdir / name, args.tau)
        if c2 is None:
            print(f'{name}: no bimodal canon cache (+alt), skipping')
            continue
        ratio = c2_alt / np.clip(c2, EPS, None)
        flag = np.abs(ratio - 1.) > 1e-4
        dense = depths[name]
        H, W = dense.shape
        gt = load_gt(gt_dir, Path(name).stem, (H, W))
        if gt is None:
            continue
        valid = np.isfinite(gt) & (gt > 0)
        # scale-align run->GT on unflagged pixels
        m = valid & ~flag
        d_al = np.median(np.log(np.clip(gt[m], EPS, None)) - np.log(np.clip(dense[m], EPS, None)))

        sel = flag & valid & (dense > 0)
        if sel.sum() < 20:
            continue
        vv, uu = np.nonzero(sel)
        uv = np.stack([uu, vv], 1).astype(np.float64)
        z_dom = dense[sel]
        z_alt = z_dom * ratio[sel]

        others = [(intrinsics[j], w2c[j], ref_depths[names[j]])
                  for j in range(len(names)) if j != a]
        v_dom = consistency_votes(uv, z_dom, intrinsics[a], cam2w[a], others, args.rel_tol)
        v_alt = consistency_votes(uv, z_alt, intrinsics[a], cam2w[a], others, args.rel_tol)
        pick_alt = v_alt > v_dom + args.margin        # near-ties -> keep dominant

        gt_l = np.log(np.clip(gt[sel], EPS, None))
        e_dom = np.abs(np.log(z_dom) + d_al - gt_l)
        e_alt = np.abs(np.log(z_alt) + d_al - gt_l)
        gt_prefers_alt = e_alt < e_dom
        decided = np.abs(v_alt - v_dom) > args.margin
        acc_i = np.mean(pick_alt[decided] == gt_prefers_alt[decided]) if decided.any() else np.nan
        e_read = np.where(pick_alt, e_alt, e_dom)

        row = dict(acc=acc_i, err_dom=np.median(e_dom), err_read=np.median(e_read),
                   err_orc=np.median(np.minimum(e_dom, e_alt)), nflag=int(sel.sum()))
        for k, v in row.items():
            acc[k].append(v)
        print(f"{name}: n={row['nflag']:6d}  readout-acc {acc_i:.1%} "
              f"(decided {decided.mean():.0%})  err dom {row['err_dom']:.3f} "
              f"-> readout {row['err_read']:.3f}  (oracle {row['err_orc']:.3f})")

    if acc['acc']:
        print(f"\n===== AGGREGATE over {len(acc['acc'])} images =====")
        print(f"  readout accuracy vs GT:   {np.nanmean(acc['acc']):.1%}")
        print(f"  dense @flagged: dominant  {np.nanmean(acc['err_dom']):.3f}")
        print(f"  dense @flagged: readout   {np.nanmean(acc['err_read']):.3f}")
        print(f"  dense @flagged: GT-oracle {np.nanmean(acc['err_orc']):.3f}")


if __name__ == '__main__':
    main()
