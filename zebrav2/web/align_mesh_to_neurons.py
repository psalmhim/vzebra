"""
Align brain_mesh.json mesh vertices to neuron positions from newcell.mat.

Uses paired landmarks (neuron region centroid <-> mesh region centroid)
to fit a similarity transform (uniform scale + rotation + translation)
from mesh space -> neuron space via SVD-based Procrustes analysis,
then applies it to ALL mesh vertices in brain_mesh.json and position.mat.

Similarity (Procrustes) is preferred over full affine because:
  - Preserves shape (no axis-dependent squash/shear)
  - Robust when Y-axis landmarks are near-midline (poorly constrained)
  - 7 DOF (scale + 3 rotation + 3 translation) vs 12 DOF affine

Run: .venv/bin/python zebrav2/web/align_mesh_to_neurons.py
"""
import os
import json
import numpy as np
from scipy.io import loadmat, savemat

BASE = os.path.dirname(__file__)
STATIC = os.path.join(BASE, 'static')

ATLAS_JSON = os.path.join(STATIC, 'brain_atlas.json')
MESH_JSON = os.path.join(STATIC, 'brain_mesh.json')
NEWCELL_MAT = os.path.join(STATIC, 'newcell.mat')
POSITION_MAT = os.path.join(STATIC, 'position.mat')

# Mapping: mesh region name -> list of neuron region names
# For bilateral regions, we average both L and R neuron subregions
LANDMARK_MAP = {
    'Cb':  ['cerebellum'],
    'TeO': ['sfgs_b_L', 'sfgs_b_R'],        # bilateral tectum
    'P':   ['pal_s', 'pal_d'],               # pallium
    'Hb':  ['habenula'],
    'PT':  ['reticulospinal'],
    'OB':  ['olfaction'],
    'Hi':  ['insula'],
    'MON': ['lateral_line'],
    'SP':  ['d1', 'd2'],                     # subpallium / basal ganglia
    'T':   ['critic'],                        # tegmentum
}


def compute_neuron_centroids(atlas_data):
    """Compute centroids for each neuron region group from brain_atlas.json."""
    x = np.array(atlas_data['x'])
    y = np.array(atlas_data['y'])
    z = np.array(atlas_data['z'])
    scale = atlas_data['scale']
    regions = atlas_data['regions']

    region_name_to_meta = {r['n']: r for r in regions}
    centroids = {}

    for mesh_name, neuron_names in LANDMARK_MAP.items():
        all_pts = []
        for nn in neuron_names:
            meta = region_name_to_meta[nn]
            off = meta['off']
            cnt = meta['count']
            pts = np.column_stack([
                x[off:off + cnt] * scale,
                y[off:off + cnt] * scale,
                z[off:off + cnt] * scale,
            ])
            all_pts.append(pts)
        all_pts = np.vstack(all_pts)
        centroids[mesh_name] = all_pts.mean(axis=0)

    return centroids


def compute_mesh_centroids(mesh_data):
    """Compute centroids for each mesh region from brain_mesh.json."""
    centroids = {}
    for region in mesh_data['regions']:
        name = region['n']
        if name in LANDMARK_MAP:
            v = np.array(region['v']).reshape(-1, 3)
            centroids[name] = v.mean(axis=0)
    return centroids


def fit_similarity(src_pts, dst_pts):
    """
    Fit similarity transform: dst = scale * (R @ src) + t
    using SVD-based Procrustes analysis.

    src_pts, dst_pts: (N, 3) arrays of paired landmarks
    Returns scale (float), R (3,3), t (3,)
    """
    src_c = src_pts.mean(axis=0)
    dst_c = dst_pts.mean(axis=0)
    M = src_pts - src_c
    N = dst_pts - dst_c

    # Cross-covariance
    H = M.T @ N
    U, S, Vt = np.linalg.svd(H)

    # Ensure proper rotation (det > 0)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, np.sign(d)])
    R = Vt.T @ D @ U.T

    # Uniform scale
    scale = np.trace(R @ H) / np.trace(M.T @ M)

    # Translation
    t = dst_c - scale * (R @ src_c)

    return scale, R, t


def apply_similarity(pts, scale, R, t):
    """Apply similarity: result = scale * (pts @ R.T) + t"""
    return scale * (pts @ R.T) + t


def main():
    print("=" * 60)
    print("MESH-TO-NEURON ALIGNMENT (Procrustes similarity)")
    print("=" * 60)

    # ── Step 0: Undo previous bad affine if detected ─────────────────────
    # The bad affine was applied once; we invert it to restore original data.
    # Detection: check if TeO mesh Y-range < 200 (original ~418, corrupted ~182)
    with open(MESH_JSON) as f:
        mesh_data = json.load(f)

    for region in mesh_data['regions']:
        if region['n'] == 'TeO':
            v = np.array(region['v']).reshape(-1, 3)
            y_range = v[:, 1].max() - v[:, 1].min()
            break

    if y_range < 300:
        print(f"\nDetected corrupted mesh (TeO Y-range={y_range:.1f}, expected ~418)")
        print("Inverting previous bad affine transform...")

        # Reconstruct the exact bad affine that was applied
        bad_mesh_centroids = np.array([
            [58.4, 4.3, 84.8],       # Cb
            [-229.5, -6.1, 55.9],    # Hb
            [-34.6, -0.9, -98.9],    # Hi
            [145.4, 0.6, 102.9],     # MON
            [-338.0, 0.8, -45.1],    # OB
            [-269.0, 0.8, -11.8],    # P
            [-116.8, -0.3, -65.8],   # PT
            [-286.9, 2.0, -78.6],    # SP
            [-22.8, 2.1, -5.9],      # T
            [-63.3, -0.2, 70.0],     # TeO
        ])
        bad_neuron_centroids = np.array([
            [130.5, -1.1, 47.3],     # Cb
            [-163.7, -9.2, -7.3],    # Hb
            [41.5, 6.8, -159.7],     # Hi
            [225.2, -0.2, 94.7],     # MON
            [-293.6, 0.3, -140.8],   # OB
            [-229.7, -5.9, -72.8],   # P
            [-48.2, 0.8, -137.2],    # PT
            [-234.1, 1.4, -140.5],   # SP
            [58.5, 3.9, -73.6],      # T
            [28.5, -2.0, 39.4],      # TeO
        ])
        n = bad_mesh_centroids.shape[0]
        X = np.hstack([bad_mesh_centroids, np.ones((n, 1))])
        P, _, _, _ = np.linalg.lstsq(X, bad_neuron_centroids, rcond=None)
        A_bad = P[:3, :].T
        b_bad = P[3, :]

        A_inv = np.linalg.inv(A_bad)
        b_inv = -b_bad @ A_inv.T

        # Invert brain_mesh.json
        for region in mesh_data['regions']:
            v = np.array(region['v']).reshape(-1, 3)
            v_restored = v @ A_inv.T + b_inv
            region['v'] = [round(float(x), 2) for x in v_restored.flatten()]

        with open(MESH_JSON, 'w') as f:
            json.dump(mesh_data, f, separators=(',', ':'))

        # Invert position.mat
        pos = loadmat(POSITION_MAT)
        mesh_count = int(pos['mesh_count'].flatten()[0])
        for i in range(1, mesh_count + 1):
            key = f'mesh_vertices_{i}'
            v = pos[key]
            pos[key] = v @ A_inv.T + b_inv
        savemat(POSITION_MAT, {k: v for k, v in pos.items()
                                if not k.startswith('_')},
                do_compression=True)

        # Verify restoration
        for region in mesh_data['regions']:
            if region['n'] == 'TeO':
                v = np.array(region['v']).reshape(-1, 3)
                y_range_new = v[:, 1].max() - v[:, 1].min()
                print(f"  Restored TeO Y-range: {y_range_new:.1f} (should be ~418)")
                break

        # Reload
        with open(MESH_JSON) as f:
            mesh_data = json.load(f)

    # ── Step 1: Load neuron positions ─────────────────────────────────────
    with open(ATLAS_JSON) as f:
        atlas_data = json.load(f)

    neuron_centroids = compute_neuron_centroids(atlas_data)
    mesh_centroids = compute_mesh_centroids(mesh_data)

    # Build paired landmark arrays
    names = sorted(LANDMARK_MAP.keys())
    src_pts = np.array([mesh_centroids[n] for n in names])    # mesh space
    dst_pts = np.array([neuron_centroids[n] for n in names])  # neuron space

    print(f"\nLandmark pairs ({len(names)} regions):")
    print(f"{'Region':>6s}  {'Mesh centroid':>30s}  "
          f"{'Neuron centroid':>30s}  {'Offset':>25s}")
    for i, name in enumerate(names):
        m = src_pts[i]
        n = dst_pts[i]
        d = n - m
        print(f"{name:>6s}  ({m[0]:7.1f},{m[1]:7.1f},{m[2]:7.1f})  "
              f"({n[0]:7.1f},{n[1]:7.1f},{n[2]:7.1f})  "
              f"({d[0]:+6.1f},{d[1]:+6.1f},{d[2]:+6.1f})")

    # ── Step 2: Pre-alignment error ──────────────────────────────────────
    pre_errors = np.linalg.norm(dst_pts - src_pts, axis=1)
    print(f"\nPre-alignment RMS error:  {np.sqrt(np.mean(pre_errors**2)):.2f}")
    print(f"Pre-alignment max error:  {pre_errors.max():.2f}")
    print(f"Pre-alignment mean error: {pre_errors.mean():.2f}")

    # ── Step 3: Fit similarity transform ─────────────────────────────────
    scale, R, t = fit_similarity(src_pts, dst_pts)

    print(f"\nSimilarity transform parameters:")
    print(f"  Scale: {scale:.6f}")
    print(f"  Rotation matrix:")
    for row in R:
        print(f"    [{row[0]:+9.6f} {row[1]:+9.6f} {row[2]:+9.6f}]")
    print(f"  Translation: [{t[0]:+8.3f} {t[1]:+8.3f} {t[2]:+8.3f}]")

    # Rotation angle
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    print(f"  Rotation angle: {angle:.2f} degrees")
    det = np.linalg.det(R)
    print(f"  det(R): {det:.6f} ({'proper' if det > 0 else 'improper'} rotation)")

    # ── Step 4: Post-alignment error on landmarks ────────────────────────
    transformed_src = apply_similarity(src_pts, scale, R, t)
    post_errors = np.linalg.norm(dst_pts - transformed_src, axis=1)
    print(f"\nPost-alignment RMS error:  {np.sqrt(np.mean(post_errors**2)):.2f}")
    print(f"Post-alignment max error:  {post_errors.max():.2f}")
    print(f"Post-alignment mean error: {post_errors.mean():.2f}")

    print(f"\nPer-region post-alignment errors:")
    for i, name in enumerate(names):
        print(f"  {name:>6s}: {post_errors[i]:.2f}")

    # ── Step 5: Apply transform to brain_mesh.json ───────────────────────
    print("\nApplying similarity transform to brain_mesh.json...")
    total_verts = 0
    for region in mesh_data['regions']:
        v = np.array(region['v']).reshape(-1, 3)
        v_new = apply_similarity(v, scale, R, t)
        region['v'] = [round(float(x), 2) for x in v_new.flatten()]
        total_verts += len(v)
    print(f"  Transformed {total_verts} vertices across "
          f"{len(mesh_data['regions'])} regions")

    with open(MESH_JSON, 'w') as f:
        json.dump(mesh_data, f, separators=(',', ':'))
    size_kb = os.path.getsize(MESH_JSON) / 1024
    print(f"  Saved: {MESH_JSON} ({size_kb:.0f} KB)")

    # ── Step 6: Apply transform to position.mat ──────────────────────────
    print("\nApplying similarity transform to position.mat...")
    pos = loadmat(POSITION_MAT)
    mesh_count = int(pos['mesh_count'].flatten()[0])
    total_mat_verts = 0
    for i in range(1, mesh_count + 1):
        key = f'mesh_vertices_{i}'
        v = pos[key]  # (N, 3)
        v_new = apply_similarity(v, scale, R, t)
        pos[key] = v_new
        total_mat_verts += v.shape[0]
    print(f"  Transformed {total_mat_verts} vertices across {mesh_count} meshes")

    savemat(POSITION_MAT, {k: v for k, v in pos.items()
                            if not k.startswith('_')},
            do_compression=True)
    size_kb = os.path.getsize(POSITION_MAT) / 1024
    print(f"  Saved: {POSITION_MAT} ({size_kb:.0f} KB)")

    # ── Step 7: Verify ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    with open(MESH_JSON) as f:
        mesh_check = json.load(f)
    new_mesh_centroids = compute_mesh_centroids(mesh_check)

    print(f"\n{'Region':>6s}  {'New mesh centroid':>30s}  "
          f"{'Neuron centroid':>30s}  {'Error':>8s}")
    for name in names:
        m = new_mesh_centroids[name]
        n = neuron_centroids[name]
        err = np.linalg.norm(m - n)
        print(f"{name:>6s}  ({m[0]:7.1f},{m[1]:7.1f},{m[2]:7.1f})  "
              f"({n[0]:7.1f},{n[1]:7.1f},{n[2]:7.1f})  {err:8.2f}")

    # Verify shape preservation (TeO bilateral Y-range)
    for region in mesh_check['regions']:
        if region['n'] == 'TeO':
            v = np.array(region['v']).reshape(-1, 3)
            print(f"\nTeO shape check:")
            print(f"  X: [{v[:,0].min():.1f}, {v[:,0].max():.1f}]  "
                  f"range={v[:,0].max()-v[:,0].min():.1f}")
            print(f"  Y: [{v[:,1].min():.1f}, {v[:,1].max():.1f}]  "
                  f"range={v[:,1].max()-v[:,1].min():.1f}")
            print(f"  Z: [{v[:,2].min():.1f}, {v[:,2].max():.1f}]  "
                  f"range={v[:,2].max()-v[:,2].min():.1f}")
            break

    print("\nDone.")


if __name__ == '__main__':
    main()
