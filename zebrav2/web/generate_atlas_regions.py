"""
Generate atlas_regions.json for the web Atlas tab.
Reads MPIN subject-12 cell positions (72-region bilateral voxel atlas) + region labels,
subsamples to ≤800 pts/region, and writes JSON for the Three.js atlas3d.js viewer.

Coordinate space: voxel×10 with scale=0.1 (matches brain_atlas.json and brain_mesh.json).

Run: .venv/bin/python -m zebrav2.web.generate_atlas_regions
"""
import json, csv, ast, numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
ATLAS = ROOT / "atlas"
OUT  = ROOT / "zebrav2/web/static/atlas_regions.json"

MAX_PTS = 800   # max neurons per region to keep rendering fast

# Region number → abbreviation (72 bilateral: 0-35 left, 36-71 right)
RNUM_TO_ABBR = {
    0:'MON', 1:'Cb', 2:'MOS1', 3:'MOS2', 4:'MOS3', 5:'MOS4', 6:'MOS5',
    7:'IPN', 8:'IO', 9:'Hc', 10:'Ra', 11:'T', 12:'aRF', 13:'imRF', 14:'pRF',
    15:'GG', 16:'Hb', 17:'Hi', 18:'Hr', 19:'OG', 20:'OB', 21:'OE',
    22:'P', 23:'Pi', 24:'PT', 25:'PO', 26:'PrT', 27:'R', 28:'SP', 29:'TeO',
    30:'Th', 31:'TL', 32:'TS', 33:'TG', 34:'VR', 35:'NX',
}


def load_colors():
    """Returns list of [r,g,b] per label (index = grey_level 0..35)."""
    color_path = ATLAS / "colormap_roinames.json"
    with open(color_path) as f:
        d = json.load(f)
    cmap = d["colormap"]
    colors = []
    for i in range(36):
        if i < len(cmap):
            r, g, b = cmap[i][0], cmap[i][1], cmap[i][2]
        else:
            r, g, b = 0.5, 0.5, 0.5
        colors.append([round(r, 4), round(g, 4), round(b, 4)])
    return colors


def main():
    print("Loading 72-region bilateral atlas (subject_12_CellXYZ.csv) …")
    xyz = np.loadtxt(ATLAS / "subject_12_CellXYZ.csv",
                     delimiter=',', skiprows=1, dtype=np.float32)
    region_nums = []
    with open(ATLAS / "subject_12_region_num.csv") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            region_nums.append(ast.literal_eval(row[0])[0])
    region_nums = np.array(region_nums, dtype=np.int32)

    assert len(xyz) == len(region_nums)
    colors = load_colors()

    regions = []
    rng = np.random.default_rng(42)

    # Combine left (k) and right (k+36) into single combined regions (0-35)
    for gl in range(36):
        abbr = RNUM_TO_ABBR.get(gl, f"R{gl}")
        mask_L = region_nums == gl
        mask_R = region_nums == (gl + 36)
        mask = mask_L | mask_R
        pts = xyz[mask]
        if len(pts) == 0:
            continue

        total_count = int(mask.sum())

        # Subsample
        if len(pts) > MAX_PTS:
            idx = rng.choice(len(pts), MAX_PTS, replace=False)
            pts = pts[idx]

        c = colors[gl] if gl < len(colors) else [0.5, 0.5, 0.5]

        # Store as voxel×10 (brain_atlas.json format: scale 0.1 → voxels)
        regions.append({
            "label": gl,
            "name":  abbr,
            "abbr":  abbr,
            "count": total_count,
            "c":     c,
            "x": [round(float(v) * 10, 1) for v in pts[:, 0]],
            "y": [round(float(v) * 10, 1) for v in pts[:, 1]],
            "z": [round(float(v) * 10, 1) for v in pts[:, 2]],
        })

    out = {
        "scale": 0.1,
        "regions": regions,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, separators=(",", ":"))

    size_kb = OUT.stat().st_size / 1024
    print(f"Wrote {OUT}  ({size_kb:.0f} KB, {len(regions)} regions)")


if __name__ == "__main__":
    main()
