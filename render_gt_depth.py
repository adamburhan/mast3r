#!/usr/bin/env python3
"""Render GT depth for ETH3D undistorted DSLR images by raycasting the scan mesh.

Uses the COLMAP-format calibration (dslr_calibration_undistorted, PINHOLE) and
the occlusion surface mesh, both in the ETH3D world frame. Writes one float32
.npy per image: z-depth in the camera frame, NaN where the mesh is not hit,
proportionally downscaled to --long_side so downstream loaders can nearest-
resize to any smaller working resolution.

Usage:
  python render_gt_depth.py --scene ~/scratch/datasets/eth3d/kicker
  # -> <scene>/gt_depth_render/DSC_6490.npy ...

Requires numpy + open3d. Run --selftest (numpy only) to check the pose/ray math.
"""
import argparse
from pathlib import Path

import numpy as np


def quat_to_R(qw, qx, qy, qz):
    q = np.array([qw, qx, qy, qz], float)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def read_colmap_cameras(fname):
    cams = {}
    for line in open(fname):
        if line.startswith('#') or not line.strip():
            continue
        el = line.split()
        cam_id, model = int(el[0]), el[1]
        assert model == 'PINHOLE', f'expected PINHOLE (undistorted calib), got {model}'
        fx, fy, cx, cy = map(float, el[4:8])
        cams[cam_id] = dict(W=int(el[2]), H=int(el[3]), fx=fx, fy=fy, cx=cx, cy=cy)
    return cams


def read_colmap_images(fname):
    # images.txt alternates: pose line / 2D-points line (possibly empty)
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


def make_rays(cam, R, t, long_side):
    """World-frame rays for a proportionally downscaled pixel grid.
    Directions have unit z in the camera frame, so t_hit IS the z-depth."""
    s = long_side / max(cam['W'], cam['H'])
    Ws, Hs = round(cam['W'] * s), round(cam['H'] * s)
    fx, fy, cx, cy = (cam[k] * s for k in ('fx', 'fy', 'cx', 'cy'))
    j, i = np.meshgrid(np.arange(Ws), np.arange(Hs))
    dirs_cam = np.stack([(j + 0.5 - cx) / fx, (i + 0.5 - cy) / fy,
                         np.ones((Hs, Ws))], axis=-1).reshape(-1, 3)
    origin = -R.T @ t
    dirs_w = dirs_cam @ R          # rows: (R^T d)^T
    rays = np.concatenate([np.broadcast_to(origin, dirs_w.shape), dirs_w],
                          axis=1).astype(np.float32)
    return rays, (Hs, Ws)


def render_scene(scene_dir, mesh_path, calib_dir, out_dir, long_side):
    import open3d as o3d
    cams = read_colmap_cameras(calib_dir / 'cameras.txt')
    views = read_colmap_images(calib_dir / 'images.txt')
    print(f'{len(views)} views, {len(cams)} camera(s), mesh: {mesh_path}')

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    assert len(mesh.triangles) > 0, f'empty mesh: {mesh_path}'
    rc = o3d.t.geometry.RaycastingScene()
    rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    out_dir.mkdir(exist_ok=True)
    for v in views:
        rays, (Hs, Ws) = make_rays(cams[v['cam_id']], v['R'], v['t'], long_side)
        t_hit = rc.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
        depth = np.where(np.isfinite(t_hit) & (t_hit > 0), t_hit,
                         np.nan).reshape(Hs, Ws).astype(np.float32)
        np.save(out_dir / (Path(v['name']).stem + '.npy'), depth)
        hit = np.isfinite(depth)
        print(f'  {v["name"]}: {Hs}x{Ws}, hit {hit.mean():.1%}, '
              f'z [{np.nanmin(depth):.2f}, {np.nanmax(depth):.2f}]')
    print(f'wrote {len(views)} depth maps to {out_dir}')


def selftest():
    # quaternion convention: identity and 90 deg about z (COLMAP w,x,y,z)
    assert np.allclose(quat_to_R(1, 0, 0, 0), np.eye(3))
    R90 = quat_to_R(np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4))
    assert np.allclose(R90 @ [1, 0, 0], [0, 1, 0], atol=1e-12), R90

    # rays: identity pose, principal pixel looks down +z, t_hit == z-depth
    cam = dict(W=640, H=480, fx=500., fy=500., cx=320., cy=240.)
    rays, (Hs, Ws) = make_rays(cam, np.eye(3), np.zeros(3), 64)
    assert (Hs, Ws) == (48, 64)
    center = rays[(Hs // 2) * Ws + Ws // 2]
    assert np.allclose(center[:3], 0)
    assert np.allclose(center[3:], [0.5 / 50., 0.5 / 50., 1.], atol=1e-6), center
    assert np.allclose(rays[:, 5], 1.), 'unit z in cam frame -> t_hit is z-depth'

    # posed camera at (0,0,-2) looking at origin: origin recovered
    t = np.array([0., 0., 2.])     # x_cam = R x_w + t -> center at -R^T t = (0,0,-2)
    rays, _ = make_rays(cam, np.eye(3), t, 64)
    assert np.allclose(rays[0, :3], [0, 0, -2])

    # parser round-trip on a synthetic images.txt with empty points2D lines
    import tempfile, os
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
        f.write('# comment\n1 1 0 0 0 0.5 -1 2 1 dslr_images_undistorted/DSC_0002.JPG\n\n'
                '2 1 0 0 0 0 0 0 1 dslr_images_undistorted/DSC_0001.JPG\n'
                '1 2 3 4 -1\n')
        fname = f.name
    views = read_colmap_images(fname)
    os.unlink(fname)
    assert [v['name'] for v in views] == ['DSC_0001.JPG', 'DSC_0002.JPG']
    assert np.allclose(views[1]['t'], [0.5, -1, 2])
    print('selftest OK')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path, help='ETH3D scene dir (e.g. .../eth3d/kicker)')
    p.add_argument('--mesh', type=Path, default=None,
                   help='default: <scene>/occlusion/surface_mesh.ply')
    p.add_argument('--calib', type=Path, default=None,
                   help='default: <scene>/dslr_calibration_undistorted')
    p.add_argument('--out', type=Path, default=None,
                   help='default: <scene>/gt_depth_render')
    p.add_argument('--long_side', type=int, default=1520)
    p.add_argument('--selftest', action='store_true')
    args = p.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.scene:
        p.error('--scene is required (or use --selftest)')
    render_scene(args.scene,
                 args.mesh or args.scene / 'occlusion' / 'surface_mesh.ply',
                 args.calib or args.scene / 'dslr_calibration_undistorted',
                 args.out or args.scene / 'gt_depth_render',
                 args.long_side)


if __name__ == '__main__':
    main()
