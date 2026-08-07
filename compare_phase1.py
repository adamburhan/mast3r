#!/usr/bin/env python3
"""Phase-1 comparison: baseline vs bimodal canonical mode, against rendered GT.

For each image: the flag mask is recovered by diffing the two canonical caches
(pixels where bimodal changed the depth), then scale-aligned log-depth error is
reported for (a) the canonical maps themselves and (b) the final post-BA dense
depth of both runs, stratified by flagged / unflagged pixels.

Usage:
  python compare_phase1.py --scene ~/scratch/datasets/eth3d/kicker \
      --cache ~/scratch/mast3r_out/baseline/cache/kicker \
      --run_a ~/scratch/mast3r_out/baseline/kicker \
      --run_b ~/scratch/mast3r_out/bimodal_p1/kicker \
      --tau 0.15
(assumes GT at <scene>/gt_depth_render, from render_gt_depth.py)
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mast3r.utils.misc import hash_md5

EPS = 1e-8


def load_canon(cache_dir, img_path, mode, subsample=8):
    kw = {'mode': mode}
    f = cache_dir / 'canon_views' / (hash_md5(str(img_path)) + f'_{subsample=}_{kw=}.pth')
    if not f.exists():
        return None
    (canon, canon2, cconf), focal = torch.load(f, map_location='cpu', weights_only=False)
    return canon.numpy()


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


def aligned_logerr(z, gt_l, valid):
    zl = np.log(np.clip(z, EPS, None))
    d = np.median((gt_l - zl)[valid])
    return np.abs(zl + d - gt_l)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True, help='dir containing canon_views/')
    p.add_argument('--run_a', type=Path, required=True, help='baseline out dir (depth_*.npy)')
    p.add_argument('--run_b', type=Path, required=True, help='bimodal out dir (depth_*.npy)')
    p.add_argument('--gt_dir', type=Path, default=None, help='default: <scene>/gt_depth_render')
    p.add_argument('--tau', type=float, default=0.15)
    args = p.parse_args()
    gt_dir = args.gt_dir or args.scene / 'gt_depth_render'

    imgs = sorted((args.scene / 'images' / 'dslr_images_undistorted').glob('*.JPG'))
    acc = {k: [] for k in ('flag_frac', 'canon_a_f', 'canon_b_f', 'dense_a_f',
                           'dense_b_f', 'dense_a_u', 'dense_b_u', 'dense_a_all', 'dense_b_all')}

    for img in imgs:
        canon_a = load_canon(args.cache, img, 'avg-angle')
        canon_b = load_canon(args.cache, img, f'bimodal-{args.tau}')
        da_f, db_f = args.run_a / f'depth_{img.stem}.npy', args.run_b / f'depth_{img.stem}.npy'
        if canon_a is None or canon_b is None or not (da_f.exists() and db_f.exists()):
            print(f'{img.name}: missing inputs '
                  f'(canon a/b: {canon_a is not None}/{canon_b is not None}, '
                  f'dense a/b: {da_f.exists()}/{db_f.exists()})')
            continue

        za, zb = canon_a[..., 2], canon_b[..., 2]
        flag = np.abs(zb / np.clip(za, EPS, None) - 1.) > 1e-4
        H, W = za.shape

        gt = load_gt(gt_dir, img.stem, (H, W))
        if gt is None:
            print(f'{img.name}: no GT, skipping')
            continue
        gt_l = np.log(np.clip(gt, EPS, None))
        valid = np.isfinite(gt) & (gt > 0)

        dense_a, dense_b = np.load(da_f), np.load(db_f)
        errs = {name: aligned_logerr(z, gt_l, valid)
                for name, z in (('ca', za), ('cb', zb), ('da', dense_a), ('db', dense_b))}

        fl, un = flag & valid, ~flag & valid
        if fl.sum() < 20:
            print(f'{img.name}: flag {flag.mean():.2%} (<20 valid flagged px, skipping stats)')
            continue
        row = dict(flag_frac=flag.mean(),
                   canon_a_f=np.median(errs['ca'][fl]), canon_b_f=np.median(errs['cb'][fl]),
                   dense_a_f=np.median(errs['da'][fl]), dense_b_f=np.median(errs['db'][fl]),
                   dense_a_u=np.median(errs['da'][un]), dense_b_u=np.median(errs['db'][un]),
                   dense_a_all=np.median(errs['da'][valid]), dense_b_all=np.median(errs['db'][valid]))
        for k, v in row.items():
            acc[k].append(v)
        print(f"{img.name}: flag {row['flag_frac']:.2%}  "
              f"canon flagged {row['canon_a_f']:.3f}->{row['canon_b_f']:.3f}  "
              f"DENSE flagged {row['dense_a_f']:.3f}->{row['dense_b_f']:.3f}  "
              f"unflagged {row['dense_a_u']:.3f}->{row['dense_b_u']:.3f}")

    if acc['flag_frac']:
        print(f"\n===== AGGREGATE over {len(acc['flag_frac'])} images "
              f"(mean of per-image medians, |log z err|) =====")
        print(f"  flagged fraction:        {np.mean(acc['flag_frac']):.2%}")
        print(f"  canonical @flagged:      {np.mean(acc['canon_a_f']):.3f} -> {np.mean(acc['canon_b_f']):.3f}")
        print(f"  post-BA dense @flagged:  {np.mean(acc['dense_a_f']):.3f} -> {np.mean(acc['dense_b_f']):.3f}")
        print(f"  post-BA dense @unflagged:{np.mean(acc['dense_a_u']):.3f} -> {np.mean(acc['dense_b_u']):.3f}")
        print(f"  post-BA dense @all:      {np.mean(acc['dense_a_all']):.3f} -> {np.mean(acc['dense_b_all']):.3f}")


if __name__ == '__main__':
    main()
