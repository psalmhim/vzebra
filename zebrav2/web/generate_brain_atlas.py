"""
Generate brain atlas JSON using the 72-region bilateral zebrafish atlas.

Atlas: subject_12_CellXYZ.csv + subject_12_region_num.csv
  - 47010 cells with (x,y,z) in voxel space
  - 72 regions: 0-35 left hemisphere (prefix 'l'), 36-71 right hemisphere (prefix 'r')
  - Coordinate axes: x=anterior→posterior, y=left→right (midline ~310), z=ventral→dorsal
  - Connectome: subject_12_sc_list_pre_post.csv, indices 0–47009

Region numbering (left: 0-35, right: +36):
  0/36:MON  1/37:Cb  2-6/38-42:MOS1-5  7/43:IPN  8/44:IO  10/46:Ra  11/47:T
  12/48:aRF 13/49:imRF 14/50:pRF  16/52:Hb  17/53:Hi  19/55:OG  20/56:OB
  21/57:OE  22/58:P  24/60:PT  25/61:PO  26/62:PrT  28/64:SP  29/65:TeO
  30/66:Th  31/67:TL  32/68:TS  33/69:TG  34/70:VR  35/71:NX
  (missing in s12: 9/45:Hc, 15/51:GG, 18/54:Hr, 23/59:Pi, 27/63:R)

Each brain module entry:
  'atlas': list of region numbers to average for centroid ([] = use 'pos')
  'side':  'L'|'R' = use that side's atlas centroid; 'mid' = average L+R
  'pos':   [x,y,z] voxel fallback when atlas cells are absent
  'spread': Gaussian spread in voxels (default 15)
  'x/y/z_shift': sublayer offset in voxels
"""
import os, json, csv, ast
import numpy as np

ATLAS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'atlas')
OUT_PATH  = os.path.join(os.path.dirname(__file__), 'static', 'brain_atlas.json')
# Authoritative neuron positions — always use this file when it exists
NEWCELL_PATH = os.path.join(os.path.dirname(__file__), 'static', 'newcell.mat')

# Retina (regions 27/63) is missing in s12 — manual voxel position
# Eye cups are anterior (small x), very lateral (small y for left, large for right)
_RETINA_L_POS = [220, 115, 165]   # left eye (very lateral, y small)
_RETINA_R_POS = [220, 505, 165]   # right eye (very lateral, y large)

REGIONS = {
    # ── RETINA ─────────────────────────────────────────────────────────────
    # Atlas region 27 (lR) / 63 (rR) missing in s12 — use estimated voxel pos
    'retina_L':       {'n': 1000, 'atlas': [],    'pos': _RETINA_L_POS, 'spread': 25, 'color': '#aaccff'},
    'retina_R':       {'n': 1000, 'atlas': [],    'pos': _RETINA_R_POS, 'spread': 25, 'color': '#ffccaa'},

    # ── TECTUM = TeO (29=lTeO, 65=rTeO) — four depth layers ──────────────
    # z_shift separates SFGS-b / SFGS-d / SGC / SO layers dorsoventrally
    'sfgs_b_L':       {'n': 1200, 'atlas': [29],  'side': 'L', 'spread': 20, 'z_shift':  18, 'color': '#00cccc'},
    'sfgs_b_R':       {'n': 1200, 'atlas': [65],  'side': 'R', 'spread': 20, 'z_shift':  18, 'color': '#00eeee'},
    'sfgs_d_L':       {'n': 1200, 'atlas': [29],  'side': 'L', 'spread': 20, 'z_shift':   5, 'color': '#00aaaa'},
    'sfgs_d_R':       {'n': 1200, 'atlas': [65],  'side': 'R', 'spread': 20, 'z_shift':   5, 'color': '#00bbbb'},
    'sgc_L':          {'n':  400, 'atlas': [29],  'side': 'L', 'spread': 15, 'z_shift':  -8, 'color': '#cc4444'},
    'sgc_R':          {'n':  400, 'atlas': [65],  'side': 'R', 'spread': 15, 'z_shift':  -8, 'color': '#dd5555'},
    'so_L':           {'n':  400, 'atlas': [29],  'side': 'L', 'spread': 15, 'z_shift': -20, 'color': '#8844cc'},
    'so_R':           {'n':  400, 'atlas': [65],  'side': 'R', 'spread': 15, 'z_shift': -20, 'color': '#9955dd'},

    # ── THALAMUS = Th (30=lTh, 66=rTh) ───────────────────────────────────
    'tc_L':           {'n':  300, 'atlas': [30],  'side': 'L', 'spread': 12, 'color': '#cccc00'},
    'tc_R':           {'n':  300, 'atlas': [66],  'side': 'R', 'spread': 12, 'color': '#dddd00'},
    'trn_L':          {'n':   80, 'atlas': [30],  'side': 'L', 'spread':  8, 'z_shift': 8, 'color': '#888800'},
    'trn_R':          {'n':   80, 'atlas': [66],  'side': 'R', 'spread':  8, 'z_shift': 8, 'color': '#999900'},

    # ── PALLIUM = P (22=lP, 58=rP) ────────────────────────────────────────
    'pal_s':          {'n': 1600, 'atlas': [22, 58], 'spread': 25, 'color': '#44cc00'},
    'pal_d':          {'n':  800, 'atlas': [22, 58], 'spread': 20, 'z_shift': -12, 'color': '#2266ff'},
    'place_cells':    {'n':  128, 'atlas': [22, 58], 'spread': 12, 'x_shift':  15, 'color': '#88ff88'},
    'working_memory': {'n':   40, 'atlas': [22, 58], 'spread':  8, 'x_shift':  25, 'color': '#99ff99'},

    # ── SUBPALLIUM = SP (28=lSP, 64=rSP) — basal ganglia + amygdala ──────
    'd1':             {'n':  400, 'atlas': [28, 64], 'spread': 10, 'color': '#ffaa00'},
    'd2':             {'n':  300, 'atlas': [28, 64], 'spread': 10, 'x_shift':  8, 'color': '#00aaff'},
    'gpi':            {'n':   60, 'atlas': [28, 64], 'spread':  6, 'x_shift': -8, 'color': '#cc8800'},
    'amygdala':       {'n':   50, 'atlas': [28, 64], 'spread':  6, 'z_shift':-12, 'color': '#ff4444'},

    # ── HABENULA = Hb (16=lHb, 52=rHb) ───────────────────────────────────
    'habenula':       {'n':   50, 'atlas': [16, 52], 'spread':  8, 'color': '#ff6644'},

    # ── CEREBELLUM = Cb (1=lCb, 37=rCb) ──────────────────────────────────
    'cerebellum':     {'n':  270, 'atlas': [ 1, 37], 'spread': 20, 'color': '#cc66ff'},

    # ── HYPOTHALAMUS = Hi (17=lHi, 53=rHi) ───────────────────────────────
    'insula':         {'n':   34, 'atlas': [17, 53], 'spread':  8, 'color': '#ff88cc'},
    'allostasis':     {'n':   30, 'atlas': [17, 53], 'spread':  7, 'x_shift': 10, 'color': '#ee6699'},
    'circadian':      {'n':    6, 'atlas': [17, 53], 'spread':  5, 'x_shift':-15, 'color': '#ffee88'},
    'sleep_wake':     {'n':    4, 'atlas': [17, 53], 'spread':  4, 'x_shift':-22, 'color': '#88eeff'},

    # ── TEGMENTUM = T (11=lT, 47=rT) ─────────────────────────────────────
    'critic':         {'n':   68, 'atlas': [11, 47], 'spread': 10, 'color': '#44cc44'},
    'da':             {'n':   60, 'atlas': [11, 47], 'spread':  8, 'x_shift':-10, 'color': '#f39c12'},
    'ach':            {'n':   30, 'atlas': [11, 47], 'spread':  7, 'x_shift': 10, 'color': '#27ae60'},

    # ── POSTERIOR RF = pRF (14=lpRF, 50=rpRF) ────────────────────────────
    'habit':          {'n':   40, 'atlas': [14, 50], 'spread':  8, 'color': '#aa8844'},
    'na':             {'n':   30, 'atlas': [14, 50], 'spread':  7, 'x_shift': -8, 'color': '#2980b9'},

    # ── RAPHE = Ra (10=lRa, 46=rRa) ───────────────────────────────────────
    'serotonin':      {'n':   40, 'atlas': [10, 46], 'spread': 10, 'color': '#8e44ad'},

    # ── PRETECTUM = PrT (26=lPrT, 62=rPrT) ───────────────────────────────
    'predictive':     {'n':  192, 'atlas': [26, 62], 'spread': 18, 'color': '#66aaff'},
    'saccade':        {'n':    6, 'atlas': [26, 62], 'spread':  5, 'x_shift': 8, 'color': '#ffaaff'},

    # ── POSTERIOR TUBERCULUM = PT (24=lPT, 60=rPT) ───────────────────────
    'reticulospinal': {'n':   42, 'atlas': [24, 60], 'spread':  8, 'color': '#ff8800'},

    # ── VAGUS MOTOR NEURONS = NX (35=lNX, 71=rNX) — most posterior ───────
    'cpg_L':          {'n':   48, 'atlas': [35],  'side': 'L', 'spread': 10, 'color': '#ffaa44'},
    'cpg_R':          {'n':   48, 'atlas': [71],  'side': 'R', 'spread': 10, 'color': '#ffbb55'},

    # ── CLASSIFIER (tectum-anchored, dorsal to sfgs_b) ────────────────────
    'classifier':     {'n':  128, 'atlas': [29, 65], 'spread': 15, 'z_shift': 28, 'color': '#88cccc'},

    # ── GOAL SELECTOR = PO/preoptic (25=lPO, 61=rPO) ─────────────────────
    'goal':           {'n':    4, 'atlas': [25, 61], 'spread':  5, 'color': '#ffffff'},

    # ── LATERAL LINE + VESTIBULAR = MON (0=lMON, 36=rMON) ────────────────
    'lateral_line':   {'n':   24, 'atlas': [ 0, 36], 'spread':  8, 'color': '#44ffaa'},
    'vestibular':     {'n':    6, 'atlas': [ 0, 36], 'spread':  5, 'z_shift': 8, 'color': '#ffff44'},

    # ── OLFACTION = OB (20=lOB, 56=rOB) ──────────────────────────────────
    'olfaction':      {'n':   28, 'atlas': [20, 56], 'spread':  8, 'color': '#ff99ff'},

    # ── PROPRIOCEPTION — spinal/hindbrain (NX vicinity) ───────────────────
    'proprioception': {'n':    8, 'atlas': [35, 71], 'spread':  5, 'x_shift': 5, 'color': '#ff8844'},

    # ── COLOR VISION (tectum, above SFGS-b layer) ─────────────────────────
    'color_vision':   {'n':   32, 'atlas': [29, 65], 'spread': 12, 'z_shift': 32, 'color': '#ff44ff'},
}


def main():
    # ── Load 72-region bilateral atlas ─────────────────────────────────────
    xyz_csv = np.loadtxt(os.path.join(ATLAS_DIR, 'subject_12_CellXYZ.csv'),
                         delimiter=',', skiprows=1, dtype=np.float32)
    region_nums = []
    with open(os.path.join(ATLAS_DIR, 'subject_12_region_num.csv')) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            region_nums.append(ast.literal_eval(row[0])[0])
    region_nums = np.array(region_nums, dtype=np.int32)

    assert len(xyz_csv) == len(region_nums), \
        f'Cell count mismatch: xyz={len(xyz_csv)} vs labels={len(region_nums)}'
    n_atlas_cells = len(xyz_csv)
    print(f'Atlas: {n_atlas_cells} cells, {len(np.unique(region_nums))} regions present')

    # Precompute centroid per region number
    centroid_cache = {}
    for rnum in np.unique(region_nums):
        pts = xyz_csv[region_nums == rnum]
        centroid_cache[int(rnum)] = pts.mean(axis=0)

    def get_centroid(region):
        """Average centroid of a list of atlas region numbers."""
        pts = [centroid_cache[r] for r in region if r in centroid_cache]
        if not pts:
            return None
        return np.mean(pts, axis=0).astype(np.float32)

    rng = np.random.default_rng(42)
    region_names = list(REGIONS.keys())

    all_x, all_y, all_z, all_ri = [], [], [], []

    for ridx, rname in enumerate(region_names):
        info   = REGIONS[rname]
        n      = info['n']
        atlas  = info['atlas']
        spread = info.get('spread', 15.0)

        if atlas:
            c = get_centroid(atlas)
            if c is None:
                # Fallback: all zeros (won't happen with well-defined regions)
                c = np.zeros(3, dtype=np.float32)
                print(f'  WARNING: no atlas cells for {rname} (regions {atlas})')
        else:
            c = np.array(info['pos'], dtype=np.float32)

        c = c.copy()
        c[0] += info.get('x_shift', 0.0)
        c[1] += info.get('y_shift', 0.0)
        c[2] += info.get('z_shift', 0.0)

        positions = rng.normal(loc=c, scale=spread, size=(n, 3)).astype(np.float32)

        for i in range(n):
            all_x.append(float(positions[i, 0]))
            all_y.append(float(positions[i, 1]))
            all_z.append(float(positions[i, 2]))
            all_ri.append(ridx)

    N = len(all_x)
    print(f'Model neurons: {N}')

    # --- Override positions from newcell.mat if available ---
    if os.path.exists(NEWCELL_PATH):
        from scipy.io import loadmat
        newcell = loadmat(NEWCELL_PATH)['neuron_xyz']
        assert newcell.shape[0] == N, \
            f'newcell.mat has {newcell.shape[0]} neurons but model has {N}'
        all_x = newcell[:, 0].tolist()
        all_y = newcell[:, 1].tolist()
        all_z = newcell[:, 2].tolist()
        print(f'Using neuron positions from newcell.mat ({N} neurons)')
        # Store as float (already in final coordinates)
        qx = [round(v, 2) for v in all_x]
        qy = [round(v, 2) for v in all_y]
        qz = [round(v, 2) for v in all_z]
        scale = 1.0
    else:
        # Store as ×10 integers (0.1-voxel precision)
        qx = [int(round(v * 10)) for v in all_x]
        qy = [int(round(v * 10)) for v in all_y]
        qz = [int(round(v * 10)) for v in all_z]
        scale = 0.1

    region_meta = []
    offset = 0
    for rname in region_names:
        n = REGIONS[rname]['n']
        region_meta.append({'n': rname, 'c': REGIONS[rname]['color'],
                            'count': n, 'off': offset})
        offset += n

    # ── Connectome ─────────────────────────────────────────────────────────
    print('Loading connectome...')
    conn_path = os.path.join(ATLAS_DIR, 'subject_12_sc_list_pre_post.csv')
    raw_conn  = np.loadtxt(conn_path, delimiter=',', skiprows=1, dtype=np.int32)

    # Map atlas region number → model region indices
    rnum_to_ridxs = {}
    for ridx, rname in enumerate(region_names):
        for rnum in REGIONS[rname]['atlas']:
            rnum_to_ridxs.setdefault(rnum, []).append(ridx)

    # Model region neuron ranges
    region_ranges = {}
    off = 0
    for rname in region_names:
        n = REGIONS[rname]['n']
        region_ranges[rname] = (off, off + n)
        off += n

    # Sample connections
    sample_idx = rng.choice(len(raw_conn), size=min(500_000, len(raw_conn)), replace=False)
    conn_flat = []
    conn_set  = set()
    for idx in sample_idx:
        pre, post = raw_conn[idx]
        if pre >= n_atlas_cells or post >= n_atlas_cells:
            continue
        pre_rnum  = int(region_nums[pre])
        post_rnum = int(region_nums[post])
        pre_ridxs  = rnum_to_ridxs.get(pre_rnum,  [])
        post_ridxs = rnum_to_ridxs.get(post_rnum, [])
        if not pre_ridxs or not post_ridxs:
            continue
        sri = rng.choice(pre_ridxs)
        tri = rng.choice(post_ridxs)
        s_lo, s_hi = region_ranges[region_names[sri]]
        t_lo, t_hi = region_ranges[region_names[tri]]
        s = int(rng.integers(s_lo, s_hi))
        t = int(rng.integers(t_lo, t_hi))
        key = (s, t)
        if key not in conn_set and s != t:
            conn_set.add(key)
            conn_flat.extend([s, t, sri, tri])
            if len(conn_set) >= 3000:
                break

    def ri(name):
        return region_names.index(name) if name in region_names else -1

    # Explicit retina → tectum (contralateral) — primary visual pathway
    retina_l_range = region_ranges.get('retina_L', (0, 0))
    retina_r_range = region_ranges.get('retina_R', (0, 0))
    retina_l_idx   = ri('retina_L')
    retina_r_idx   = ri('retina_R')
    tectum_L_ridxs = [ri(n) for n in ['sfgs_b_L','sfgs_d_L','sgc_L','so_L'] if ri(n) >= 0]
    tectum_R_ridxs = [ri(n) for n in ['sfgs_b_R','sfgs_d_R','sgc_R','so_R'] if ri(n) >= 0]
    all_tec_ridxs  = tectum_L_ridxs + tectum_R_ridxs

    for _ in range(200):
        if retina_l_idx >= 0 and tectum_R_ridxs:  # L eye → R tectum (contralateral)
            s   = int(rng.integers(*retina_l_range))
            tri = rng.choice(tectum_R_ridxs)
            conn_flat.extend([s, int(rng.integers(*region_ranges[region_names[tri]])),
                              retina_l_idx, int(tri)])
        if retina_r_idx >= 0 and tectum_L_ridxs:  # R eye → L tectum (contralateral)
            s   = int(rng.integers(*retina_r_range))
            tri = rng.choice(tectum_L_ridxs)
            conn_flat.extend([s, int(rng.integers(*region_ranges[region_names[tri]])),
                              retina_r_idx, int(tri)])

    # Tectum → pallium / thalamus hierarchy
    pal_ridx_val = ri('pal_s')
    tc_L_ridx    = ri('tc_L')
    tc_R_ridx    = ri('tc_R')
    for _ in range(100):
        src_tri = rng.choice(all_tec_ridxs)
        s = int(rng.integers(*region_ranges[region_names[src_tri]]))
        if pal_ridx_val >= 0 and rng.random() < 0.5:
            conn_flat.extend([s, int(rng.integers(*region_ranges['pal_s'])),
                              int(src_tri), pal_ridx_val])
        else:
            tc_choice = tc_L_ridx if src_tri in tectum_L_ridxs else tc_R_ridx
            if tc_choice >= 0:
                conn_flat.extend([s, int(rng.integers(*region_ranges[region_names[tc_choice]])),
                                  int(src_tri), tc_choice])

    n_conn = len(conn_flat) // 4
    print(f'Connections: {n_conn}')

    output = {
        'N': N, 'x': qx, 'y': qy, 'z': qz,
        'scale': scale,
        'ri': [int(v) for v in all_ri],
        'regions': region_meta,
        'conn': [int(v) for v in conn_flat],
        'units': 'voxels',
    }

    with open(OUT_PATH, 'w') as f:
        json.dump(output, f, separators=(',', ':'))

    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f'Saved: {OUT_PATH} ({size_kb:.0f} KB)')


if __name__ == '__main__':
    main()
