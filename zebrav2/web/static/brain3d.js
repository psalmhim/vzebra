// Brain 3D — Three.js WebGL, real micron coordinates, optimized

async function initBrain3D() {
    if (brain3d.initialized) return;
    const container = document.getElementById('brain3dContainer');
    const canvas = document.getElementById('brain3dCanvas');
    if (!container || !canvas || container.clientWidth < 50 || container.clientHeight < 50) {
        setTimeout(initBrain3D, 500); return;
    }

    let raw;
    try { raw = await (await fetch('/static/brain_atlas.json?v=' + Date.now())).json(); }
    catch(e) { document.getElementById('brain3dInfo').textContent = 'Failed'; return; }

    const N = raw.N, S = raw.scale || 0.1;
    const px = raw.x, py = raw.y, pz = raw.z, ri = raw.ri;
    const regions = raw.regions;
    brain3d.atlas = { N, px, py, pz, ri, regions, connections: [], S };
    const cf = raw.conn || [];
    for (let i = 0; i < cf.length; i += 4)
        brain3d.atlas.connections.push({s:cf[i], t:cf[i+1], sri:cf[i+2], tri:cf[i+3]});
    for (const r of regions) brain3d.regionOffsets[r.n] = { start: r.off, count: r.count };

    const w = container.clientWidth, h = container.clientHeight;
    const regionRGB = regions.map(r => {
        const v = parseInt(r.c.replace('#',''), 16);
        return [(v>>16&255)/255, (v>>8&255)/255, (v&255)/255];
    });

    // Scene — dark background, very light fog (don't darken distant neurons)
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x2a1a0a);
    scene.fog = new THREE.Fog(0x2a1a0a, 1500, 4000);  // linear fog: starts far, gentle fade
    brain3d.scene = scene;

    // Camera
    const cam = new THREE.PerspectiveCamera(45, w/h, 1, 5000);
    cam.position.set(0, 200, 800);
    brain3d.camera = cam;

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    brain3d.renderer = renderer;

    const controls = new THREE.OrbitControls(cam, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.15;
    controls.target.set(0, 0, 0);
    brain3d.controls = controls;

    // Lighting — bright enough to see gray neurons clearly
    scene.add(new THREE.AmbientLight(0x667788, 1.0));
    // Key light — warm white from upper-right
    const key = new THREE.DirectionalLight(0xffeedd, 0.8);
    key.position.set(400, 500, 300); scene.add(key);
    // Fill light — cool blue from lower-left
    const fill = new THREE.DirectionalLight(0x6699cc, 0.4);
    fill.position.set(-300, -100, -400); scene.add(fill);
    // Rim/back light
    const rim = new THREE.DirectionalLight(0x88aaff, 0.4);
    rim.position.set(-200, 300, -500); scene.add(rim);

    // --- InstancedMesh: neurons ---
    const geo = new THREE.SphereGeometry(3, 6, 4);
    const mat = new THREE.MeshPhongMaterial({
        transparent: true, opacity: 0.85,
        shininess: 60, specular: 0x556677,
        emissive: 0x000000,
    });
    const mesh = new THREE.InstancedMesh(geo, mat, N);
    brain3d.mesh = mesh;

    brain3d.baseColors = new Float32Array(N * 3);
    brain3d.spikeDecay = new Float32Array(N);
    brain3d._spikeFrame = 0;
    const dummy = new THREE.Object3D();
    const col = new THREE.Color();

    for (let i = 0; i < N; i++) {
        dummy.position.set(px[i]*S, py[i]*S, pz[i]*S);  // PCA-aligned, direct XYZ
        dummy.scale.setScalar(0.4);  // inactive = small
        dummy.updateMatrix();
        mesh.setMatrixAt(i, dummy.matrix);
        const rgb = regionRGB[ri[i]];
        brain3d.baseColors[i*3]=rgb[0]; brain3d.baseColors[i*3+1]=rgb[1]; brain3d.baseColors[i*3+2]=rgb[2];
        // Inactive = desaturated hint of region color (gray + 15% region tint)
        col.setRGB(0.45 + rgb[0]*0.15, 0.45 + rgb[1]*0.15, 0.45 + rgb[2]*0.15);
        mesh.setColorAt(i, col);
    }
    mesh.instanceMatrix.needsUpdate = true;
    mesh.instanceColor.needsUpdate = true;
    scene.add(mesh);

    // Subtle axes
    const axes = new THREE.AxesHelper(80);
    axes.material.opacity = 0.15; axes.material.transparent = true;
    scene.add(axes);

    // Connections toggle
    const connEl = document.getElementById('brain3dShowAllConn');
    if (connEl) connEl.onchange = (e) => {
        if (e.target.checked) showAllConnections();
        else clearConnections();
    };

    document.getElementById('brain3dInfo').innerHTML =
        `<span style="color:#58a6ff">${N.toLocaleString()}</span> neurons · ` +
        `<span style="color:#7ee787">${regions.length}</span> regions · ` +
        `<span style="color:#f0883e">${brain3d.atlas.connections.length}</span> conn · microns`;

    // Load brain surface mesh
    loadBrainMesh(scene);

    brain3d.initialized = true;
    animate();
}

async function loadBrainMesh(scene) {
    let meshData;
    try { meshData = await (await fetch('/static/brain_mesh.json?v=' + Date.now())).json(); }
    catch(e) { console.warn('brain_mesh.json not found'); return; }

    const S = meshData.scale || 0.1;
    console.log('brain_mesh loaded:', meshData.regions.length, 'regions, scale=', S);
    const group = new THREE.Group();

    for (const region of meshData.regions) {
        const vRaw = region.v;
        const fRaw = region.f;
        const nVerts = vRaw.length / 3;
        const nFaces = fRaw.length / 3;
        if (nVerts < 3 || nFaces < 1) continue;

        const positions = new Float32Array(nVerts * 3);
        for (let i = 0; i < nVerts; i++) {
            positions[i*3]   = vRaw[i*3]   * S;  // PCA-aligned, direct XYZ
            positions[i*3+1] = vRaw[i*3+1] * S;
            positions[i*3+2] = vRaw[i*3+2] * S;
        }
        const indices = new Uint32Array(fRaw);

        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geo.setIndex(new THREE.BufferAttribute(indices, 1));
        geo.computeVertexNormals();

        const c = region.c;
        const mat = new THREE.MeshPhongMaterial({
            color: new THREE.Color(c[0], c[1], c[2]),
            transparent: true,
            opacity: region.a || 0.12,
            side: THREE.DoubleSide,
            depthWrite: false,
        });
        group.add(new THREE.Mesh(geo, mat));
    }
    scene.add(group);
    brain3d.meshGroup = group;
    console.log('brain_mesh group added:', group.children.length, 'meshes');
}

// --- Connections ---
function showAllConnections() {
    clearConnections();
    const a = brain3d.atlas, sc = a.S;
    const conns = a.connections;
    const positions = new Float32Array(conns.length * 6);
    for (let i = 0; i < conns.length; i++) {
        const c = conns[i];
        positions[i*6]   = a.px[c.s]*sc; positions[i*6+1] = a.py[c.s]*sc; positions[i*6+2] = a.pz[c.s]*sc;
        positions[i*6+3] = a.px[c.t]*sc; positions[i*6+4] = a.py[c.t]*sc; positions[i*6+5] = a.pz[c.t]*sc;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.LineBasicMaterial({ color: 0x99ccff, transparent: true, opacity: 0.6 });
    const lines = new THREE.LineSegments(geo, mat);
    brain3d.connGroup = new THREE.Group();
    brain3d.connGroup.add(lines);
    const glow = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({
        color: 0x6699cc, transparent: true, opacity: 0.2,
    }));
    glow.scale.set(1.002, 1.002, 1.002);
    brain3d.connGroup.add(glow);
    brain3d.scene.add(brain3d.connGroup);
}

function clearConnections() {
    if (brain3d.connGroup) {
        brain3d.scene.remove(brain3d.connGroup);
        brain3d.connGroup.traverse(c => {
            if (c.geometry) c.geometry.dispose();
            if (c.material) c.material.dispose();
        });
        brain3d.connGroup = null;
    }
}

// --- Spike updates ---
function updateBrain3DSpikes(spikes) {
    if (!brain3d.mesh || !brain3d.atlas || !brain3d.spikeDecay) return;
    if (!document.getElementById('brain3dSpikes')?.checked) return;

    // Keys match BrainV2 trainer output (15 regions from real brain)
    // + retina from idle demo. Unknown keys are silently ignored.
    const spikeMap = {
        // === Real brain keys (from engine/trainer.py) ===
        'sfgs_b':    ['sfgs_b_L', 'sfgs_b_R'],   // tectum superficial
        'sfgs_d':    ['sfgs_d_L', 'sfgs_d_R'],   // tectum deep
        'sgc':       ['sgc_L',    'sgc_R'],       // stratum griseum centrale
        'so':        ['so_L',     'so_R'],        // stratum opticum
        'tc':        ['tc_L',     'tc_R'],        // thalamus TC
        'trn':       ['trn_L',    'trn_R'],       // thalamus TRN
        'pal_s':     ['pal_s'],                   // pallium superficial
        'pal_d':     ['pal_d'],                   // pallium deep
        'd1':        ['d1'],                      // striatum D1
        'd2':        ['d2'],                      // striatum D2
        'amygdala':  ['amygdala'],
        'cerebellum':['cerebellum'],
        'habenula':  ['habenula'],
        'critic':    ['critic'],
        'insula':    ['insula'],
        // === Extra keys from idle demo ===
        'retina':    ['retina_L', 'retina_R'],   // combined (from real brain)
        'retina_L':  ['retina_L'],                // per-eye (from idle demo)
        'retina_R':  ['retina_R'],                // per-eye (from idle demo)
        'place_cells':    ['place_cells'],
        'predictive':     ['predictive'],
        'lateral_line':   ['lateral_line'],
        'olfaction':      ['olfaction'],
        'cpg':            ['cpg_L', 'cpg_R'],
        'reticulospinal': ['reticulospinal'],
    };

    const decay = brain3d.spikeDecay;
    const col = new THREE.Color();

    // Decay toward inactive
    for (let i = 0; i < decay.length; i++) decay[i] *= 0.75;

    // Inject spikes
    for (const [key, rnList] of Object.entries(spikeMap)) {
        const act = spikes[key] || 0;
        if (act < 0.05) continue;
        const intensity = Math.min(1.0, Math.log1p(act) / 4.0);
        for (const rn of rnList) {
            const off = brain3d.regionOffsets[rn];
            if (!off) continue;
            for (let i = off.start; i < off.start + off.count; i++) {
                if (Math.random() < intensity * 0.8) {
                    decay[i] = Math.min(1.0, decay[i] + 0.5 + Math.random() * 0.5);
                }
            }
        }
    }

    // Apply: desaturated gray-tint (g=0) → vivid bright region color (g=1)
    // Size:  0.4 (silent) → 1.3 (firing) for strong pop-out
    const dummy = new THREE.Object3D();
    const a = brain3d.atlas, S = a.S;
    for (let i = 0; i < decay.length; i++) {
        const g = decay[i];
        const r0 = brain3d.baseColors[i*3], g0 = brain3d.baseColors[i*3+1], b0 = brain3d.baseColors[i*3+2];
        // Inactive base: gray with subtle region tint
        const ir = 0.45 + r0 * 0.15, ig = 0.45 + g0 * 0.15, ib = 0.45 + b0 * 0.15;
        // Active target: boosted saturated region color
        const ar = Math.min(1.0, r0 * 1.8), ag = Math.min(1.0, g0 * 1.8), ab = Math.min(1.0, b0 * 1.8);
        col.setRGB(ir + g * (ar - ir), ig + g * (ag - ig), ib + g * (ab - ib));
        brain3d.mesh.setColorAt(i, col);
        const sc = 0.4 + 0.9 * g;
        dummy.position.set(a.px[i]*S, a.py[i]*S, a.pz[i]*S);
        dummy.scale.setScalar(sc);
        dummy.updateMatrix();
        brain3d.mesh.setMatrixAt(i, dummy.matrix);
    }
    brain3d.mesh.instanceColor.needsUpdate = true;
    brain3d.mesh.instanceMatrix.needsUpdate = true;
}

function animate() {
    requestAnimationFrame(animate);
    if (!brain3d.renderer) return;
    brain3d.controls.update();

    const container = document.getElementById('brain3dContainer');
    if (container) {
        const w = container.clientWidth, h = container.clientHeight;
        if (w > 10 && h > 10) {
            const pr = brain3d.renderer.getPixelRatio();
            const cw = brain3d.renderer.domElement.width, ch = brain3d.renderer.domElement.height;
            if (cw !== w*pr || ch !== h*pr) {
                brain3d.renderer.setSize(w, h);
                brain3d.camera.aspect = w / h;
                brain3d.camera.updateProjectionMatrix();
            }
        }
    }
    brain3d.renderer.render(brain3d.scene, brain3d.camera);
}

setTimeout(initBrain3D, 500);
