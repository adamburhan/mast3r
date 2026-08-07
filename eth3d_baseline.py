#!/usr/bin/env python3
"""MASt3R-SfM baseline on one ETH3D scene: run + pose metrics vs GT calibration.

Reproduces the paper's ETH3D protocol (Table 3): full undistorted DSLR image
set, retrieval-20-10 scene graph, shared intrinsics, paper hyperparameters
(lr1=0.07, lr2=0.014, 300+300 iters). Reports pairwise RRA@tau / RTA@tau and
mAA(30) against dslr_calibration_undistorted, and saves poses + dense depth
for later GT-depth-stratified evaluation.

Usage:
  python eth3d_baseline.py --scene ~/scratch/datasets/eth3d/kicker --out_dir out/baseline
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mast3r.model import AsymmetricMASt3R
from mast3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import load_images

torch.serialization.add_safe_globals([argparse.Namespace])


def quat_to_R(qw, qx, qy, qz):
    q = np.array([qw, qx, qy, qz], float)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def read_colmap_images(fname):
    # COLMAP images.txt alternates: pose line / 2D-points line (possibly empty)
    views = []
    expect_pose = True
    for line in open(fname):
        if line.startswith('#'):
            continue
        if expect_pose:
            if not line.strip():
                continue
            el = line.split()
            views.append(dict(R=quat_to_R(*map(float, el[1:5])),   # world->cam
                              t=np.array(list(map(float, el[5:8]))),
                              cam_id=int(el[8]), name=Path(el[9]).name))
            expect_pose = False
        else:
            expect_pose = True
    return sorted(views, key=lambda v: v['name'])


def pairwise_pose_metrics(R_w2c_est, C_est, R_w2c_gt, C_gt, taus=(1, 3, 5, 10, 15, 30)):
    """RRA/RTA over all image pairs (invariant to global similarity transform).
    Rotation: geodesic angle between relative rotations. Translation: angle
    between relative translation directions in camera-i frame."""
    N = len(C_est)
    rra, rta = [], []
    for i in range(N):
        for j in range(i + 1, N):
            dR = (R_w2c_est[i] @ R_w2c_est[j].T) @ (R_w2c_gt[i] @ R_w2c_gt[j].T).T
            rra.append(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1., 1.))))
            te = R_w2c_est[i] @ (C_est[j] - C_est[i])
            tg = R_w2c_gt[i] @ (C_gt[j] - C_gt[i])
            ne, ng = np.linalg.norm(te), np.linalg.norm(tg)
            if ne < 1e-9 or ng < 1e-9:
                rta.append(0.0)   # degenerate baseline: count as correct
            else:
                rta.append(np.degrees(np.arccos(np.clip(te @ tg / (ne * ng), -1., 1.))))
    rra, rta = np.array(rra), np.array(rta)
    m = {f'RRA@{t}': float((rra < t).mean()) for t in taus}
    m.update({f'RTA@{t}': float((rta < t).mean()) for t in taus})
    worst = np.maximum(rra, rta)
    m['mAA(30)'] = float(np.mean([(worst < t).mean() for t in range(1, 31)]))
    m['n_pairs'] = len(rra)
    return m, rra, rta


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', type=Path, required=True, help='ETH3D scene dir')
    p.add_argument('--out_dir', type=Path, default=Path('out/baseline'))
    p.add_argument('--cache_dir', type=Path, default=None,
                   help='default: <out_dir>/cache/<scene_name>')
    p.add_argument('--model', default='naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric')
    p.add_argument('--retrieval_ckpt', default='checkpoints/'
                   'MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth')
    p.add_argument('--scene_graph', default='retrieval-20-10')
    p.add_argument('--lr1', type=float, default=0.07)
    p.add_argument('--niter1', type=int, default=300)
    p.add_argument('--lr2', type=float, default=0.014)   # paper 5.1 (repo default is 0.01)
    p.add_argument('--niter2', type=int, default=300)
    p.add_argument('--matching_conf_thr', type=float, default=5.0)
    p.add_argument('--kinematic_mode', default='hclust-ward',
                   help="'hclust-ward' (repo default) or 'mst' (paper-era)")
    p.add_argument('--canonical_mode', default='avg-angle',
                   help="'avg-angle' (baseline) or 'bimodal-<tau>' e.g. 'bimodal-0.15'")
    p.add_argument('--seed', type=int, default=0,
                   help='seeds the FPS keyframe draw in retrieval (graph.py uses '
                        'unseeded np.random.choice) and torch, for reproducible graphs')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    name = args.scene.name
    out = args.out_dir / name
    out.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir or (args.out_dir / 'cache' / name)

    imgdir = args.scene / 'images' / 'dslr_images_undistorted'
    filelist = sorted(str(pth) for pth in imgdir.glob('*.JPG'))
    print(f'{name}: {len(filelist)} images')

    model = AsymmetricMASt3R.from_pretrained(args.model).to(args.device).eval()
    imgs = load_images(filelist, size=512, verbose=False)

    if args.scene_graph.startswith('retrieval'):
        from mast3r.retrieval.processor import Retriever
        retriever = Retriever(args.retrieval_ckpt, backbone=model, device=args.device)
        with torch.no_grad():
            sim_matrix = retriever(filelist)
        del retriever
        torch.cuda.empty_cache()
        pairs = make_pairs(imgs, scene_graph=args.scene_graph, symmetrize=True,
                           sim_mat=sim_matrix)
    else:
        pairs = make_pairs(imgs, scene_graph=args.scene_graph, symmetrize=True)
    print(f'{len(pairs)} pairs ({args.scene_graph})')

    scene = sparse_global_alignment(
        filelist, pairs, str(cache), model,
        lr1=args.lr1, niter1=args.niter1, lr2=args.lr2, niter2=args.niter2,
        matching_conf_thr=args.matching_conf_thr, shared_intrinsics=True,
        kinematic_mode=args.kinematic_mode, canonical_mode=args.canonical_mode,
        device=args.device)

    # ---- save outputs for later depth-stratified evaluation ----------------
    cam2w = scene.get_im_poses().cpu().numpy()
    intrinsics = torch.stack([k for k in scene.intrinsics]).cpu().numpy() \
        if isinstance(scene.intrinsics, (list, tuple)) else scene.intrinsics.cpu().numpy()
    _, dense_depth, _ = scene.get_dense_pts3d(clean_depth=False)
    np.savez_compressed(out / 'result.npz',
                        names=[Path(f).name for f in filelist],
                        cam2w=cam2w, intrinsics=intrinsics)
    for f, d, im in zip(filelist, dense_depth, scene.imgs):
        H, W = im.shape[:2]
        np.save(out / f'depth_{Path(f).stem}.npy',
                d.cpu().numpy().reshape(H, W).astype(np.float32))

    # ---- pose metrics vs COLMAP GT -----------------------------------------
    views = read_colmap_images(args.scene / 'dslr_calibration_undistorted' / 'images.txt')
    gt = {v['name']: v for v in views}
    missing = [f for f in filelist if Path(f).name not in gt]
    assert not missing, f'images without GT pose: {missing[:3]}...'
    order = [Path(f).name for f in filelist]
    R_gt = np.stack([gt[n]['R'] for n in order])                  # world->cam
    C_gt = np.stack([-gt[n]['R'].T @ gt[n]['t'] for n in order])  # centers
    R_est = np.stack([np.linalg.inv(cam2w[i][:3, :3]) for i in range(len(order))])
    C_est = cam2w[:, :3, 3]

    metrics, rra, rta = pairwise_pose_metrics(R_est, C_est, R_gt, C_gt)
    metrics['scene'] = name
    metrics['n_images'] = len(order)
    metrics['config'] = {k: str(v) for k, v in vars(args).items()}
    np.savez_compressed(out / 'pose_errors.npz', rra=rra, rta=rta, names=order)
    with open(out / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{name}: RRA@5 {metrics['RRA@5']:.1%}  RTA@5 {metrics['RTA@5']:.1%}  "
          f"mAA(30) {metrics['mAA(30)']:.1%}   (paper avg: RRA@5 81.2 RTA@5 79.7)")


if __name__ == '__main__':
    main()
