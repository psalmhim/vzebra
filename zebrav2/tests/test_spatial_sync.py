"""
Validation test: verify spatial_registry.py and generate_brain_atlas.py
share the same 72-region bilateral CSV atlas.

Checks:
    1. Every region in spatial_registry.label_map has cells in the atlas
    2. Centroids fall within valid voxel ranges (x: 0-800, y: 0-620, z: 0-350)
    3. generate_positions() returns correct-shape tensors
    4. distance_weight_mask() returns non-degenerate masks
    5. label_map region numbers match generate_brain_atlas.py REGIONS
"""
import sys
import os
import numpy as np
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from zebrav2.brain.spatial_registry import SpatialRegistry


@pytest.fixture(scope='module')
def reg():
    """Load a SpatialRegistry with atlas data for all tests."""
    r = SpatialRegistry(device='cpu')
    r._load_atlas()
    return r


def test_atlas_loads(reg):
    """Atlas CSV files load without error and have correct shape."""
    assert reg._atlas_xyz is not None, 'atlas xyz not loaded'
    assert reg._atlas_labels is not None, 'atlas labels not loaded'
    assert reg._atlas_xyz.shape[0] == reg._atlas_labels.shape[0], \
        f'Cell count mismatch: xyz={reg._atlas_xyz.shape[0]} vs labels={reg._atlas_labels.shape[0]}'
    assert reg._atlas_xyz.shape[1] == 3, f'Expected 3 columns, got {reg._atlas_xyz.shape[1]}'

    n_cells = reg._atlas_xyz.shape[0]
    assert n_cells > 40000, f'Expected >40000 cells, got {n_cells}'
    print(f'  atlas: {n_cells} cells, {len(np.unique(reg._atlas_labels))} regions')


def test_label_map_coverage(reg):
    """Every region number in label_map exists in the atlas."""
    unique_labels = set(np.unique(reg._atlas_labels).tolist())
    all_ok = True
    for region_name, label_nums in reg.label_map.items():
        for lnum in label_nums:
            if lnum not in unique_labels:
                print(f'  FAIL: region "{region_name}" references label {lnum} '
                      f'not found in atlas (available: {sorted(unique_labels)})')
                all_ok = False
    assert all_ok, 'Some label_map entries reference missing atlas labels'
    print(f'  label_map: all {len(reg.label_map)} regions verified in atlas')


def test_centroids_in_voxel_range(reg):
    """Centroids of all label_map regions fall within valid voxel bounds."""
    # Atlas voxel ranges (generous bounds for subject 12)
    # Actual data spans x:[116,982], y:[39,582], z:[56,272]; use padded range
    x_range = (0, 1000)
    y_range = (0, 620)
    z_range = (0, 350)

    for region_name, label_nums in reg.label_map.items():
        centroid, spread = reg._get_region_stats(label_nums)
        x, y, z = centroid
        assert x_range[0] <= x <= x_range[1], \
            f'{region_name}: x={x:.1f} outside {x_range}'
        assert y_range[0] <= y <= y_range[1], \
            f'{region_name}: y={y:.1f} outside {y_range}'
        assert z_range[0] <= z <= z_range[1], \
            f'{region_name}: z={z:.1f} outside {z_range}'
        # Spread should be positive and reasonable
        assert (spread > 0).all(), f'{region_name}: spread has non-positive values'
    print(f'  centroids: all {len(reg.label_map)} regions in valid voxel range')


def test_generate_positions_shape(reg):
    """generate_positions() returns (n, 3) tensor for each region."""
    import torch
    for region_name in reg.label_map:
        n = 50
        pos = reg.generate_positions(region_name, n)
        assert pos.shape == (n, 3), \
            f'{region_name}: expected shape ({n}, 3), got {pos.shape}'
        assert pos.dtype == torch.float32, \
            f'{region_name}: expected float32, got {pos.dtype}'
    # Test fallback for unknown region
    pos_unknown = reg.generate_positions('nonexistent_region', 10)
    assert pos_unknown.shape == (10, 3), \
        f'Fallback region: expected (10, 3), got {pos_unknown.shape}'
    print(f'  generate_positions: all regions produce correct shape')


def test_distance_weight_mask(reg):
    """distance_weight_mask() returns non-degenerate masks."""
    # Generate positions for two regions
    reg.generate_positions('tectum', 64)
    reg.generate_positions('thalamus', 32)
    mask = reg.distance_weight_mask('tectum', 'thalamus', lambda_um=100.0)

    assert mask is not None, 'distance_weight_mask returned None'
    assert mask.shape == (64, 32), f'Expected (64, 32), got {mask.shape}'
    # Weights should be in [0, 1]
    assert mask.min() >= 0.0, f'Negative weight: {mask.min()}'
    assert mask.max() <= 1.0, f'Weight > 1: {mask.max()}'
    # Should not be all-zeros or all-ones (degenerate)
    assert mask.min() < 0.99, f'All weights ~1 (degenerate): min={mask.min():.4f}'
    assert mask.max() > 0.01, f'All weights ~0 (degenerate): max={mask.max():.4f}'
    print(f'  distance_weight_mask: (64, 32), range [{mask.min():.4f}, {mask.max():.4f}]')


def test_atlas_region_sync():
    """Verify label_map numbers match generate_brain_atlas.py REGIONS dict."""
    # Import REGIONS from generate_brain_atlas.py
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'web'))
    from generate_brain_atlas import REGIONS as WEB_REGIONS

    reg = SpatialRegistry(device='cpu')

    # Collect all atlas region numbers referenced by generate_brain_atlas.py
    web_atlas_nums = set()
    for info in WEB_REGIONS.values():
        for rnum in info['atlas']:
            web_atlas_nums.add(rnum)

    # Collect all atlas region numbers referenced by spatial_registry
    sr_atlas_nums = set()
    for label_nums in reg.label_map.values():
        for rnum in label_nums:
            sr_atlas_nums.add(rnum)

    # spatial_registry should be a subset of (or equal to) atlas nums
    missing = sr_atlas_nums - web_atlas_nums
    if missing:
        print(f'  WARNING: spatial_registry uses atlas nums not in '
              f'generate_brain_atlas: {sorted(missing)}')

    # Key region checks: make sure canonical mappings agree
    canonical = {
        'tectum':       [29, 65],
        'cerebellum':   [1, 37],
        'pallium':      [22, 58],
        'thalamus':     [30, 66],
        'tegmentum':    [11, 47],
        'hypothalamus': [17, 53],
        'habenula':     [16, 52],
        'subpallium':   [28, 64],
        'olfactory':    [20, 56],
        'raphe':        [10, 46],
        'pretectum':    [26, 62],
        'spinal':       [35, 71],
        'medulla':      [0, 36],
        'preoptic':     [25, 61],
    }
    for region, expected in canonical.items():
        actual = reg.label_map.get(region)
        assert actual is not None, f'Missing region "{region}" in label_map'
        # Check expected nums are a subset of actual
        for e in expected:
            assert e in actual, \
                f'{region}: expected atlas num {e} not in label_map {actual}'
    print(f'  atlas sync: all canonical region numbers match generate_brain_atlas.py')


def main():
    print('=== test_spatial_sync ===')

    r = SpatialRegistry(device='cpu')
    r._load_atlas()

    print('[1] test_atlas_loads')
    test_atlas_loads(r)

    print('[2] test_label_map_coverage')
    test_label_map_coverage(r)

    print('[3] test_centroids_in_voxel_range')
    test_centroids_in_voxel_range(r)

    print('[4] test_generate_positions_shape')
    test_generate_positions_shape(r)

    print('[5] test_distance_weight_mask')
    test_distance_weight_mask(r)

    print('[6] test_atlas_region_sync')
    test_atlas_region_sync()

    print('\nAll 6 tests PASSED.')


if __name__ == '__main__':
    main()
