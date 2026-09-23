// Fly brain EEG cleaner. A real 2 s EEGdenoiseNet epoch is walked through the chain
// Input -> Antenna -> Brain -> Motor -> Decoder -> Output.
//
// Everything in the scene is measured data or a physical object it stands for:
//   hull.bin   the MaleCNS neuropil surface, remeshed in Blender with baked ambient occlusion
//   skel.bin   official MaleCNS neuron skeletons, drawn at full branch density
//   act_*.bin  the recorded activity of every neuron for this epoch
//   anatomy.json  each neuron's synaptic hop distance from the antenna, and the region flow
// The electrode, the chip and the monitor are built from primitives; they are the only things
// on screen that are not the fly.
(async function () {
  // the data files are rebuilt by the export scripts; the page must never mix a new binary with a
  // cached copy of the table that indexes it
  const DATA = './data/', V = '?v=' + (location.search.match(/v=(\w+)/) || [0, '1'])[1];
  const bin = (name, T) => fetch(DATA + name + V).then(r => r.arrayBuffer()).then(b => new T(b));
  const raw = name => fetch(DATA + name + V).then(r => r.arrayBuffer());
  const [meta, anat, skel, skelId, hullBuf, pos, group] = await Promise.all([
    fetch(DATA + 'meta.json' + V).then(r => r.json()), fetch(DATA + 'anatomy.json' + V).then(r => r.json()),
    bin('skel.bin', Float32Array), bin('skel_id.bin', Uint32Array), raw('hull.bin'),
    bin('pos.bin', Float32Array), bin('group.bin', Uint8Array),
  ]);
  const acts = await Promise.all(meta.epochs.map((_, k) => bin(`act_${k}.bin`, Uint8Array)));
  const OUTSIDE = new Set(['ME', 'LO', 'LOP', 'LA', 'AME', 'CV-anterior', 'CRN']);
  const N = meta.n, T = meta.t, neurons = anat.neurons, S = neurons.length;
  const regions = anat.regions.map((r, k) => ({ ...r, k, out: OUTSIDE.has(r.label) }));
  const $ = id => document.getElementById(id);
  $('loading').remove();

  // ---------------------------------------------------------------- scene ----
  const view = $('view');
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.75));
  view.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 1, 14000);
  scene.add(new THREE.AmbientLight(0xffffff, 0.34));
  const key = new THREE.DirectionalLight(0xffffff, 0.62);
  const fill = new THREE.DirectionalLight(0xdce6f5, 0.22);
  scene.add(key, fill);

  const hv = new Uint32Array(hullBuf, 0, 2), HNV = hv[0], HNF = hv[1];
  let off = 8;
  const hullV = new Float32Array(hullBuf, off, HNV * 3); off += HNV * 12;
  const hullN = new Float32Array(hullBuf, off, HNV * 3); off += HNV * 12;
  const hullAO = new Uint8Array(hullBuf, off, HNV); off += HNV;
  const hullF = new Uint32Array(hullBuf.slice(off, off + HNF * 12));

  const c = [0, 0, 0];
  for (let i = 0; i < HNV; i++) { c[0] += hullV[3 * i]; c[1] += hullV[3 * i + 1]; c[2] += hullV[3 * i + 2]; }
  c[0] /= HNV; c[1] /= HNV; c[2] /= HNV;
  const toScene = a => { const o = new Float32Array(a.length); for (let i = 0; i < a.length; i += 3) { o[i] = a[i] - c[0]; o[i + 1] = -(a[i + 1] - c[1]); o[i + 2] = a[i + 2] - c[2]; } return o; };
  const flipY = a => { const o = new Float32Array(a.length); for (let i = 0; i < a.length; i += 3) { o[i] = a[i]; o[i + 1] = -a[i + 1]; o[i + 2] = a[i + 2]; } return o; };
  const V3 = a => new THREE.Vector3(a[0], a[1], a[2]);
  const hullPos = toScene(hullV);
  const bbox = new THREE.Box3().setFromArray(hullPos);
  const BW = bbox.max.x - bbox.min.x, BH = bbox.max.y - bbox.min.y;

  // ---- brain surface: glass, baked occlusion, glowing where the wave is ----
  const HOT = 8;
  const hot = Array.from({ length: HOT }, () => new THREE.Vector4(0, 0, 0, 0));
  const hullMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, side: THREE.DoubleSide,
    uniforms: { uHot: { value: hot }, uR: { value: 95.0 }, uDim: { value: 0 } },
    vertexShader: `attribute float ao; varying float vAO; varying vec3 vN; varying vec3 vP; varying vec3 vE;
      void main() { vAO = ao; vN = normalize(normalMatrix * normal); vP = position;
        vec4 mv = modelViewMatrix * vec4(position, 1.0); vE = normalize(-mv.xyz);
        gl_Position = projectionMatrix * mv; }`,
    fragmentShader: `uniform vec4 uHot[${HOT}]; uniform float uR; uniform float uDim;
      varying float vAO; varying vec3 vN; varying vec3 vP; varying vec3 vE;
      void main() {
        float fres = pow(1.0 - abs(dot(normalize(vN), vE)), 2.5);
        float glow = 0.0;
        for (int i = 0; i < ${HOT}; i++) { float d = length(vP - uHot[i].xyz); glow += uHot[i].w * exp(-d * d / (2.0 * uR * uR)); }
        glow = clamp(glow, 0.0, 1.0);
        vec3 glass = mix(vec3(0.40, 0.47, 0.60), vec3(0.82, 0.87, 0.94), vAO);
        vec3 col = mix(glass, vec3(0.98, 0.60, 0.18), glow * 0.85);
        gl_FragColor = vec4(col, (0.05 + 0.26 * fres + 0.32 * glow) * (1.0 - 0.6 * uDim));
      }`,
  });
  const hullGeom = new THREE.BufferGeometry();
  hullGeom.setAttribute('position', new THREE.BufferAttribute(hullPos, 3));
  hullGeom.setAttribute('normal', new THREE.BufferAttribute(flipY(hullN), 3));
  hullGeom.setAttribute('ao', new THREE.BufferAttribute(Float32Array.from(hullAO, v => v / 255), 1));
  hullGeom.setIndex(new THREE.BufferAttribute(hullF, 1));
  scene.add(new THREE.Mesh(hullGeom, hullMat));

  // ---- neurons: every branch of every fetched skeleton ----
  const TEX_W = 8192;
  const actData = new Uint8Array(TEX_W * 4), baseData = new Uint8Array(TEX_W * 4);
  const col = new THREE.Color();
  neurons.forEach((n, s) => {
    let h = n.body >>> 0; h = Math.imul(h ^ (h >>> 16), 0x45d9f3b); h = Math.imul(h ^ (h >>> 16), 0x45d9f3b); h ^= h >>> 16;
    col.setHSL((h % 997) / 997, 0.74, 0.40);
    baseData[4 * s] = col.r * 255; baseData[4 * s + 1] = col.g * 255; baseData[4 * s + 2] = col.b * 255;
    baseData[4 * s + 3] = n.group === 1 || n.group === 2 ? 255 : 205;
  });
  const actTex = new THREE.DataTexture(actData, TEX_W, 1, THREE.RGBAFormat), baseTex = new THREE.DataTexture(baseData, TEX_W, 1, THREE.RGBAFormat);
  baseTex.needsUpdate = true;
  const lineMat = new THREE.ShaderMaterial({
    uniforms: { act: { value: actTex }, base: { value: baseTex } }, transparent: true, depthWrite: false,
    vertexShader: `attribute float nid; uniform sampler2D act; uniform sampler2D base; varying vec4 vC;
      void main() {
        vec2 uv = vec2((nid + 0.5) / ${TEX_W}.0, 0.5);
        float a = texture2D(act, uv).r; vec4 b = texture2D(base, uv);
        vC = vec4(mix(vec3(0.60, 0.64, 0.72), b.rgb, smoothstep(0.0, 0.5, a)), b.a * (0.34 + 0.66 * a));
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }`,
    fragmentShader: `varying vec4 vC; void main() { gl_FragColor = vC; }`,
  });
  const yCut = bbox.min.y - 40, zCut = bbox.max.z + 40;     // the neck bundle leaves the brain: not drawn
  const keep = [];
  for (let i = 0; i < skel.length; i += 6) {
    if (-(skel[i + 1] - c[1]) < yCut || -(skel[i + 4] - c[1]) < yCut) continue;
    if (skel[i + 2] - c[2] > zCut || skel[i + 5] - c[2] > zCut) continue;
    keep.push(i / 6);
  }
  const sp = new Float32Array(keep.length * 6), sn = new Float32Array(keep.length * 2);
  keep.forEach((sg, j) => { for (let q = 0; q < 6; q++) sp[6 * j + q] = skel[6 * sg + q]; sn[2 * j] = skelId[2 * sg]; sn[2 * j + 1] = skelId[2 * sg + 1]; });
  const skelGeom = new THREE.BufferGeometry();
  skelGeom.setAttribute('position', new THREE.BufferAttribute(toScene(sp), 3));
  skelGeom.setAttribute('nid', new THREE.BufferAttribute(sn, 1));
  scene.add(new THREE.LineSegments(skelGeom, lineMat));

  const P = skelGeom.attributes.position.array, acc = new THREE.Vector3(); let an = 0;
  for (let i = 0; i < sn.length; i++) { const g = neurons[sn[i]].group; if (g === 1 || g === 2) { acc.x += P[3 * i]; acc.y += P[3 * i + 1]; acc.z += P[3 * i + 2]; an++; } }
  const ANT = acc.multiplyScalar(1 / an);

  // -------------------------------------------------------------- hardware ----
  // The three things that are not the fly: the electrode the EEG came off, the chip that holds the
  // trained readout, and the monitor the cleaned signal goes to. Built from primitives, to scale
  // with the brain so the fly stays the biggest object in the frame.
  const U = BH / 3.4;                                        // one "unit": the models are sized in these
  const ELEC = new THREE.Vector3(BW * 0.95, BH * 0.10, 0);
  const CHIP = new THREE.Vector3(-BW * 0.88, -BH * 0.16, 0);
  const MON = new THREE.Vector3(-BW * 1.46, -BH * 0.10, 0);
  const metal = (c, s, sh) => new THREE.MeshPhongMaterial({ color: c, specular: s, shininess: sh, flatShading: false });

  function electrode(at) {
    const g = new THREE.Group();
    const skin = new THREE.Mesh(new THREE.SphereGeometry(U * 1.5, 40, 28, 0, Math.PI * 2, 0, 0.62),
      new THREE.MeshPhongMaterial({ color: 0xe9cdb8, specular: 0x2a2018, shininess: 8, transparent: true, opacity: 0.92 }));
    skin.rotation.z = -Math.PI / 2; skin.position.x = U * 0.55;
    const cup = new THREE.Mesh(new THREE.CylinderGeometry(U * 0.42, U * 0.46, U * 0.16, 36), metal(0xd8dde4, 0x9aa3ad, 70));
    const pin = new THREE.Mesh(new THREE.CylinderGeometry(U * 0.13, U * 0.13, U * 0.30, 24), metal(0xb9c0c8, 0x8f98a2, 80));
    const gel = new THREE.Mesh(new THREE.CylinderGeometry(U * 0.30, U * 0.30, U * 0.05, 28),
      new THREE.MeshPhongMaterial({ color: 0x9fd8ff, specular: 0xffffff, shininess: 120, transparent: true, opacity: 0.75 }));
    for (const m of [cup, pin, gel]) m.rotation.z = Math.PI / 2;
    cup.position.x = U * 0.06; pin.position.x = U * 0.26; gel.position.x = -U * 0.06;
    const lead = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3([
      new THREE.Vector3(U * 0.38, 0, 0), new THREE.Vector3(U * 1.1, U * 0.5, U * 0.2),
      new THREE.Vector3(U * 1.5, U * 1.4, -U * 0.1), new THREE.Vector3(U * 1.2, U * 2.3, 0)]), 40, U * 0.055, 12),
      new THREE.MeshPhongMaterial({ color: 0x3a4250, shininess: 30 }));
    g.add(skin, cup, pin, gel, lead);
    g.position.copy(at); scene.add(g);
    return g;
  }
  function chip(at) {
    const g = new THREE.Group();
    const body = new THREE.Mesh(new THREE.BoxGeometry(U * 1.5, U * 1.5, U * 0.26),
      new THREE.MeshPhongMaterial({ color: 0x23282f, specular: 0x2e343c, shininess: 22 }));
    const lid = new THREE.Mesh(new THREE.BoxGeometry(U * 1.0, U * 1.0, U * 0.06), metal(0x3d434c, 0x7c848f, 70));
    lid.position.z = -U * 0.16;
    g.add(body, lid);
    const pinGeo = new THREE.BoxGeometry(U * 0.10, U * 0.26, U * 0.05);
    const gold = metal(0xd9b566, 0xfff0c0, 90);
    for (let i = 0; i < 12; i++) {                            // two pin rows, as on a real package
      const x = (i / 11 - 0.5) * U * 1.24;
      for (const s of [1, -1]) {
        const p = new THREE.Mesh(pinGeo, gold);
        p.position.set(x, s * U * 0.86, 0); g.add(p);
      }
    }
    g.position.copy(at); scene.add(g);
    return g;
  }
  function monitor(at) {
    const g = new THREE.Group();
    const shell = new THREE.Mesh(new THREE.BoxGeometry(U * 2.3, U * 1.6, U * 0.16),
      new THREE.MeshPhongMaterial({ color: 0x2a3038, specular: 0x3a424c, shininess: 26 }));
    const glass = new THREE.Mesh(new THREE.PlaneGeometry(U * 2.06, U * 1.36),
      new THREE.MeshBasicMaterial({ color: 0x0d1117, side: THREE.DoubleSide }));
    glass.position.z = -U * 0.09;
    const neck = new THREE.Mesh(new THREE.CylinderGeometry(U * 0.10, U * 0.13, U * 0.5, 20), metal(0x323942, 0x6a7480, 50));
    neck.position.y = -U * 1.05;
    const foot = new THREE.Mesh(new THREE.CylinderGeometry(U * 0.55, U * 0.62, U * 0.09, 32), metal(0x2a3038, 0x6a7480, 50));
    foot.position.y = -U * 1.32;
    g.add(shell, glass, neck, foot);
    g.position.copy(at); scene.add(g);
    return { g, glass };
  }
  const elecObj = electrode(ELEC), chipObj = chip(CHIP), monObj = monitor(MON);

  // the two screens that actually show the signal: the monitor face, and the chip lid
  function screen(w, h) {
    const cv = document.createElement('canvas'); cv.width = 384; cv.height = 240;
    const tex = new THREE.CanvasTexture(cv);
    return { cv, g: cv.getContext('2d'), tex, w, h };
  }
  const monScreen = screen(384, 240);
  monObj.glass.material = new THREE.MeshBasicMaterial({ map: monScreen.tex, side: THREE.DoubleSide });
  function paintScreen(sc, series, colours, widths, upto, bg) {
    const { g, cv } = sc, W = cv.width, H = cv.height;
    g.fillStyle = bg; g.fillRect(0, 0, W, H);
    g.strokeStyle = 'rgba(255,255,255,.06)'; g.lineWidth = 1;
    for (let i = 1; i < 4; i++) { const x = i * W / 4; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H); g.stroke(); }
    if (upto >= 0) {
      let m = 0; for (const s of series) for (let i = 0; i < T; i++) m = Math.max(m, Math.abs(s[i])); m = m || 1;
      series.forEach((s, j) => {
        g.strokeStyle = colours[j]; g.lineWidth = widths[j]; g.lineJoin = 'round'; g.beginPath();
        for (let i = 0; i <= upto; i++) { const x = i * W / (T - 1), y = H / 2 - s[i] / m * (H / 2 - 16); i ? g.lineTo(x, y) : g.moveTo(x, y); }
        g.stroke();
      });
    }
    sc.tex.needsUpdate = true;
  }

  // ---- the cable carries the signal in, the trace carries it out ----
  function connector(a, b, colour, bow = 0.22) {
    const mid = a.clone().lerp(b, 0.5).add(new THREE.Vector3(0, bow * a.distanceTo(b), 0));
    const pts = new THREE.QuadraticBezierCurve3(a, mid, b).getPoints(140);
    const tt = new Float32Array(pts.length);
    for (let i = 0; i < pts.length; i++) tt[i] = i / (pts.length - 1);
    const geo = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 140, U * 0.045, 8);
    const tube = new THREE.Mesh(geo, new THREE.MeshPhongMaterial({ color: 0x9aa3b0, shininess: 20, transparent: true, opacity: 0.30 }));
    scene.add(tube);
    const g = new THREE.BufferGeometry().setFromPoints(pts);
    g.setAttribute('t', new THREE.BufferAttribute(tt, 1));
    const m = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, depthTest: false,
      uniforms: { uHead: { value: -1 }, uCol: { value: new THREE.Color(colour) } },
      vertexShader: `attribute float t; uniform float uHead; varying float vA;
        void main() { float d = uHead - t; vA = (uHead < 0.0 || d < 0.0) ? 0.0 : exp(-d * d / 0.008);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: `uniform vec3 uCol; varying float vA; void main() { if (vA < 0.02) discard; gl_FragColor = vec4(uCol, vA); }`,
    });
    scene.add(new THREE.Line(g, m));
    return m;
  }
  const inWire = connector(ELEC.clone().add(new THREE.Vector3(U * 1.2, U * 2.3, 0)), ANT, 0xe04f24, 0.16);
  const outWire = connector(CHIP.clone().add(new THREE.Vector3(-U * 0.8, 0, 0)), MON.clone().add(new THREE.Vector3(U * 1.2, 0, 0)), 0x0f9d63, 0.20);

  // descending neurons into the chip
  const motorSlots = neurons.map((n, s) => s).filter(s => neurons[s].group === 3 && group[neurons[s].node] !== 255);
  const fan = [], fanId = [];
  for (let k = 0; k < motorSlots.length; k += 3) {
    const s = motorSlots[k], n = neurons[s].node, p = toScene(new Float32Array([pos[3 * n], pos[3 * n + 1], pos[3 * n + 2]]));
    fan.push(p[0], p[1], p[2], CHIP.x + U * 0.8, CHIP.y + (Math.random() - 0.5) * U * 1.3, CHIP.z + (Math.random() - 0.5) * U * 1.3);
    fanId.push(s, s);
  }
  const fanGeom = new THREE.BufferGeometry();
  fanGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(fan), 3));
  fanGeom.setAttribute('nid', new THREE.BufferAttribute(new Float32Array(fanId), 1));
  scene.add(new THREE.LineSegments(fanGeom, lineMat));

  // ---- labels: one or two words, nothing else ----
  const labelBox = $('labels'), labels = [];
  function label(text, at, cls = 'station', colour = '') {
    const d = document.createElement('div'); d.textContent = text; d.className = cls; if (colour) d.style.color = colour;
    labelBox.appendChild(d); const L = { d, at, alpha: 1 }; labels.push(L); return L;
  }
  label('Electrode', ELEC.clone().add(new THREE.Vector3(0, U * 3.0, 0)), 'station', '#e04f24');
  const antLab = label('Antenna', ANT.clone().add(new THREE.Vector3(0, -U * 0.7, 0)), 'station', '#e0921b');
  label('Decoder', CHIP.clone().add(new THREE.Vector3(0, U * 1.25, 0)), 'station', '#10141c');
  label('Output', MON.clone().add(new THREE.Vector3(0, U * 1.2, 0)), 'station', '#0f9d63');
  const regionCentre = regions.map(r => V3(toScene(new Float32Array(r.centre))));
  const flowByLabel = {};
  anat.flows.forEach(f => { const L = regions[f.dst].label; flowByLabel[L] = (flowByLabel[L] || 0) + f.w; });
  const regionLabel = {};
  Object.entries(flowByLabel).sort((a, b) => b[1] - a[1]).map(kv => kv[0]).filter(l => !OUTSIDE.has(l)).slice(0, 4)
    .forEach(l => { const r = regions.find(x => x.label === l && !x.out); if (r) regionLabel[l] = { L: label(l, regionCentre[r.k], 'region'), k: r.k }; });
  const sv = new THREE.Vector3();
  function placeLabels() {
    const w = view.clientWidth, h = view.clientHeight;
    for (const L of labels) {
      sv.copy(L.at).project(camera);
      const hidden = sv.z > 1 || Math.abs(sv.x) > 1.05 || Math.abs(sv.y) > 1.05 || L.alpha < 0.02;
      L.d.style.left = ((sv.x + 1) / 2 * w) + 'px';
      L.d.style.top = ((1 - sv.y) / 2 * h) + 'px';
      L.d.style.opacity = L.alpha;
      L.d.style.visibility = hidden ? 'hidden' : 'visible';
    }
  }

  // ---- camera ----
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.08;
  const centre = new THREE.Vector3((ELEC.x + MON.x) / 2, -BH * 0.04, 0);
  const SHOT = {
    wide: { at: centre, d: 2.25, el: 0.15, az: 0 },
    input: { at: ELEC.clone().lerp(ANT, 0.30), d: 0.95, el: 0.13, az: -0.16 },
    antenna: { at: ANT.clone().lerp(new THREE.Vector3(0, 0, 0), 0.35), d: 1.05, el: 0.09, az: -0.20 },
    brain: { at: new THREE.Vector3(0, -BH * 0.02, 0), d: 1.45, el: 0.21, az: 0.15 },
    motor: { at: new THREE.Vector3(CHIP.x * 0.45, CHIP.y + U * 0.6, 0), d: 1.55, el: 0.17, az: -0.10 },
    decoder: { at: CHIP.clone().lerp(new THREE.Vector3(0, 0, 0), 0.22), d: 1.15, el: 0.11, az: -0.05 },
    output: { at: CHIP.clone().lerp(MON, 0.62), d: 1.15, el: 0.09, az: 0 },
  };
  const SPAN = ELEC.distanceTo(MON);
  const camAt = new THREE.Vector3();
  const shotPos = (s, o) => { const d = s.d * SPAN * 0.56; return o.set(s.at.x + Math.sin(s.az) * d, s.at.y + Math.sin(s.el) * d, s.at.z - Math.cos(s.az) * Math.cos(s.el) * d); };
  let userCam = false, shotFrom = null, shotTo = SHOT.wide, shotT = 1;
  function setShot(s) { if (!s || s === shotTo) return; shotFrom = { pos: camera.position.clone(), tgt: controls.target.clone() }; shotTo = s; shotT = 0; }
  shotPos(SHOT.wide, camera.position); controls.target.copy(SHOT.wide.at);

  function resize() {
    const w = view.clientWidth, h = view.clientHeight;
    renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
    const dpr = Math.min(devicePixelRatio, 2);
    for (const id of ['noisy', 'motor', 'out']) {
      const cv = $(id), ch = id === 'motor' ? 84 : 62;
      cv.style.height = ch + 'px'; cv.width = (cv.clientWidth || 280) * dpr; cv.height = ch * dpr;
      cv.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
    }
  }
  addEventListener('resize', resize); resize();

  // ------------------------------------------------------------- activity ----
  const stats = acts.map(a => {
    const mu = new Float32Array(S), sd = new Float32Array(S);
    for (let t = 0; t < T; t++) { const o = t * N; for (let s = 0; s < S; s++) mu[s] += a[o + neurons[s].node]; }
    for (let s = 0; s < S; s++) mu[s] /= T;
    for (let t = 0; t < T; t++) { const o = t * N; for (let s = 0; s < S; s++) { const d = a[o + neurons[s].node] - mu[s]; sd[s] += d * d; } }
    for (let s = 0; s < S; s++) sd[s] = Math.sqrt(sd[s] / T);
    return { mu, sd };
  });
  const regW = regions.map(() => []);
  neurons.forEach((n, s) => { if (n.group === 1 || n.group === 2) return; for (const r in n.regions) regW[r].push([s, n.regions[r]]); });
  const HOPS = 4;
  const dev = new Float32Array(S), regAct = new Float32Array(regions.length);
  function activity(sample, k, p) {
    const a = acts[epoch], { mu, sd } = stats[epoch], o = sample * N;
    for (let s = 0; s < S; s++) {
      const n = neurons[s], ant = n.group === 1 || n.group === 2;
      const rec = ant ? a[o + n.node] / 255 : Math.min(1, Math.max(0, (a[o + n.node] - mu[s]) / (2.5 * sd[s] + 3)));
      const hop = n.hop < 0 ? HOPS : Math.min(n.hop, HOPS);
      let v = (k === 'idle' || k === 'input') ? 0.30 : 0;
      if (k === 'antenna') v = ant ? Math.min(1, p * 1.4) : 0.12;
      else if (k === 'brain') { const w = Math.min(1, Math.max(0, (p * (HOPS + 0.9) - hop) / 0.55)); v = ant ? 0.75 + 0.25 * rec : w * (0.35 + 0.65 * rec); }
      else if (k === 'motor') v = n.group === 3 ? 0.65 + 0.35 * Math.sin(Math.PI * Math.min(1, p)) : 0.24 * (0.4 + 0.6 * rec);
      else if (k === 'decoder' || k === 'output' || k === 'done') v = n.group === 3 ? 0.8 : 0.20;
      dev[s] = v; actData[4 * s] = v * 255;
    }
    actTex.needsUpdate = true;
    for (let r = 0; r < regions.length; r++) {
      if (regions[r].out || k !== 'brain') { regAct[r] = 0; continue; }
      let num = 0, den = 0;
      for (const [s, w] of regW[r]) { num += w * dev[s]; den += w; }
      regAct[r] = den ? Math.min(1, 2.4 * num / den) : 0;
    }
    const top = regions.filter(r => !r.out).sort((x, y) => regAct[y.k] - regAct[x.k]).slice(0, HOT);
    for (let i = 0; i < HOT; i++) { const r = top[i]; hot[i].set(regionCentre[r.k].x, regionCentre[r.k].y, regionCentre[r.k].z, regAct[r.k]); }
    for (const l in regionLabel) {
      const { L, k: rk } = regionLabel[l];
      L.alpha = regAct[rk] > 0.28 ? Math.min(1, (regAct[rk] - 0.28) * 4) : 0;
    }
  }

  // ----------------------------------------------------------------- plots ----
  const PL = { noisy: $('noisy'), motor: $('motor'), out: $('out') };
  const motorShow = motorSlots.slice().sort((a, b) => stats[0].sd[b] - stats[0].sd[a]).slice(0, 6);
  const amp = ss => { let m = 0; for (const s of ss) for (let i = 0; i < T; i++) m = Math.max(m, Math.abs(s[i])); return m || 1; };
  const grid = (g, W, H) => { g.strokeStyle = '#f1f4f8'; g.lineWidth = 1; for (let i = 1; i < 4; i++) { const x = i * W / 4; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H); g.stroke(); } };
  function drawSignal(cv, series, colours, widths, upto, shared) {
    const g = cv.getContext('2d'), W = cv.clientWidth, H = parseFloat(cv.style.height);
    g.clearRect(0, 0, W, H); grid(g, W, H);
    if (upto < 0) return;
    const common = shared ? amp(series) : 0;
    series.forEach((s, j) => {
      const m = common || amp([s]);
      g.strokeStyle = colours[j]; g.lineWidth = widths[j]; g.lineJoin = 'round'; g.beginPath();
      for (let i = 0; i <= upto; i++) { const x = i * W / (T - 1), y = H / 2 - s[i] / m * (H / 2 - 4); i ? g.lineTo(x, y) : g.moveTo(x, y); }
      g.stroke();
    });
    if (upto < T - 1) { const x = upto * W / (T - 1); g.strokeStyle = 'rgba(16,20,28,.35)'; g.lineWidth = 1; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H); g.stroke(); }
  }
  function drawMotor(cv, rows, upto) {
    const g = cv.getContext('2d'), W = cv.clientWidth, H = parseFloat(cv.style.height), h = H / rows.length;
    g.clearRect(0, 0, W, H); grid(g, W, H);
    if (upto < 0) return;
    g.strokeStyle = 'rgba(15,157,99,.85)'; g.lineWidth = 1;
    rows.forEach((s, j) => {
      let mu = 0; for (let i = 0; i < T; i++) mu += s[i]; mu /= T;
      let m = 0; for (let i = 0; i < T; i++) m = Math.max(m, Math.abs(s[i] - mu)); m = m || 1;
      g.beginPath();
      for (let i = 0; i <= upto; i++) { const x = i * W / (T - 1), y = h * (j + 0.5) - (s[i] - mu) / m * (h / 2 - 2); i ? g.lineTo(x, y) : g.moveTo(x, y); }
      g.stroke();
    });
  }
  let motorCache = null, motorEpoch = -1;
  const motorTrace = () => motorShow.map(s => { const a = acts[epoch], n = neurons[s].node, o = new Float32Array(T); for (let i = 0; i < T; i++) o[i] = a[i * N + n]; return o; });

  // this epoch against the measured baselines, including the shuffled-wiring control
  const BENCH = [
    { v: 0, name: 'no processing', tone: '#b9c0cc' },
    { v: 8.1, name: 'linear filter', tone: '#8e97a6' },
    { v: 10.4, name: 'shuffled wiring', tone: '#c8401f' },
    { v: 10.4, name: 'connectome', tone: '#0f9d63' },
  ];
  function drawBench(value) {
    const cv = $('bench'), g = cv.getContext('2d'), W = cv.clientWidth, H = 58, dpr = Math.min(devicePixelRatio, 2);
    if (cv.width !== W * dpr) { cv.width = W * dpr; cv.height = H * dpr; }
    g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
    const L = 6, R = W - 6, X = v => L + (v + 0.6) / 13.6 * (R - L), y = 32;
    g.strokeStyle = '#e4e8ee'; g.lineWidth = 2; g.beginPath(); g.moveTo(L, y); g.lineTo(R, y); g.stroke();
    g.font = '600 9px Inter, system-ui, sans-serif'; g.textBaseline = 'middle';
    BENCH.forEach((b, i) => {
      const x = X(b.v), up = i === 2;
      g.strokeStyle = b.tone; g.lineWidth = 2;
      g.beginPath(); g.moveTo(x, y - (up ? 8 : 0)); g.lineTo(x, y + (up ? 0 : 8)); g.stroke();
      g.fillStyle = b.tone; g.textAlign = i === 0 ? 'left' : 'center';
      g.fillText(b.name, i === 0 ? L : Math.min(x, R - 34), up ? y - 14 : y + 16);
    });
    g.fillStyle = '#10141c'; g.textAlign = 'center';
    const x = X(value);
    g.beginPath(); g.arc(x, y, 4, 0, 7); g.fill();
    g.font = '600 10px Inter, system-ui, sans-serif';
    g.fillText(`${value.toFixed(1)} dB`, Math.min(Math.max(x, L + 18), R - 18), y - 24);
  }

  // ------------------------------------------------------------ the stages ----
  const STAGE = [
    { key: 'input', name: 'Input', dur: 1700, shot: 'input' },
    { key: 'antenna', name: 'Antenna', dur: 1500, shot: 'antenna' },
    { key: 'brain', name: 'Brain', dur: 5200, shot: 'brain' },
    { key: 'motor', name: 'Motor', dur: 1900, shot: 'motor' },
    { key: 'decoder', name: 'Decoder', dur: 1500, shot: 'decoder' },
    { key: 'output', name: 'Output', dur: 1900, shot: 'output' },
  ];
  const rail = $('rail');
  const START = STAGE.reduce((a, s, j) => (a.push(j ? a[j - 1] + STAGE[j - 1].dur : 0), a), []);
  const TOTAL = START[START.length - 1] + STAGE[STAGE.length - 1].dur;
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let epoch = 1, elapsed = -1, playing = false, idx = -2, origin = 0;
  const play = at => { elapsed = at; origin = performance.now() - at; playing = true; };
  function syncButtons() {
    const running = elapsed >= 0 && elapsed < TOTAL;
    $('pause').disabled = !running;
    $('pause').textContent = playing ? 'Pause' : 'Resume';
  }
  STAGE.forEach((s, j) => {
    const d = document.createElement('button'); d.className = 'step'; d.type = 'button';
    d.innerHTML = `<div class="bar"><i></i></div><span>${s.name}</span>`;
    d.onclick = () => { elapsed = START[j]; playing = false; idx = -2; syncButtons(); };
    rail.appendChild(d); s.el = d; s.fill = d.querySelector('i');
  });
  controls.addEventListener('start', () => userCam = true);
  $('pause').onclick = () => { playing ? playing = false : play(elapsed); syncButtons(); };

  const LIVE = { input: 'c-in', antenna: 'c-ant', brain: 'c-brain', motor: 'c-motor', decoder: 'c-dec', output: 'c-out' };
  const epochBox = $('epochs'), names = ['Hard', 'Typical', 'Mild'];
  meta.epochs.forEach((e, k) => { const b = document.createElement('button'); b.textContent = names[k] || 'Epoch ' + k; b.onclick = () => setEpoch(k); epochBox.appendChild(b); });
  function setEpoch(k) {
    epoch = k; elapsed = -1; playing = false; idx = -2; motorEpoch = -1;
    [...epochBox.children].forEach((b, i) => b.classList.toggle('on', i === k));
    $('score').classList.remove('show'); rail.classList.remove('finished');
    STAGE.forEach(s => { s.el.classList.remove('on', 'done'); s.fill.style.width = '0'; });
    Object.values(LIVE).concat('c-mot').forEach(id => $(id).classList.remove('live'));
    $('send').querySelector('b').textContent = 'Send epoch';
    $('in-snr').textContent = ''; $('out-cc').textContent = '';
    userCam = false; setShot(SHOT.wide); syncButtons();
  }
  $('send').onclick = () => { play(0); idx = -2; userCam = false; $('score').classList.remove('show'); syncButtons(); };
  setEpoch(1);

  const ease = u => u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
  let last = performance.now();
  function frame(now) {
    requestAnimationFrame(frame);
    const dt = Math.min(now - last, 100); last = now;
    const e = meta.epochs[epoch];
    if (motorEpoch !== epoch) { motorCache = motorTrace(); motorEpoch = epoch; }
    if (playing) { elapsed = now - origin; if (elapsed >= TOTAL) { elapsed = TOTAL; playing = false; syncButtons(); } }

    let k = 'idle', p = 0, i = -1;
    if (elapsed >= 0) {
      let el = elapsed; i = 0;
      while (i < STAGE.length && el > STAGE[i].dur) { el -= STAGE[i].dur; i++; }
      if (i >= STAGE.length) { k = 'done'; i = STAGE.length - 1; p = 1; }
      else { k = STAGE[i].key; p = el / STAGE[i].dur; }
    }
    const started = kk => elapsed >= 0 && STAGE.findIndex(s => s.key === kk) <= i;
    const upto = kk => k === kk ? Math.floor(p * (T - 1)) : started(kk) ? T - 1 : -1;

    if (i !== idx) {
      idx = i;
      STAGE.forEach((s, j) => { s.el.classList.toggle('on', j === i && k !== 'done'); s.el.classList.toggle('done', j < i || k === 'done'); });
      rail.classList.toggle('finished', k === 'done');
      Object.values(LIVE).concat('c-mot').forEach(id => $(id).classList.remove('live'));
      if (LIVE[k]) $(LIVE[k]).classList.add('live');
      if (k === 'motor') $('c-mot').classList.add('live');
      const lab = $('send').querySelector('b');
      lab.textContent = k === 'idle' ? 'Send epoch' : k === 'done' ? 'Send again' : `${STAGE[i].name} · ${i + 1}/${STAGE.length}`;
      $('in-snr').textContent = started('input') ? `${e.snr_in.toFixed(1)} dB` : '';
      $('out-cc').textContent = started('output') ? `r ${e.cc.toFixed(2)}` : '';
      $('live').textContent = k === 'done' ? `Finished, ${e.snr_gain.toFixed(1)} decibels cleaner` : k === 'idle' ? '' : `Stage ${i + 1} of 6, ${STAGE[i].name}`;
      if (!userCam) setShot(SHOT[k === 'done' || k === 'idle' ? 'wide' : STAGE[i].shot]);
      syncButtons();
    }
    $('send').querySelector('i').style.width = elapsed < 0 ? '0' : (100 * Math.min(1, elapsed / TOTAL)) + '%';
    if (i >= 0 && k !== 'done') STAGE[i].fill.style.width = (p * 100) + '%';

    activity(k === 'brain' ? Math.floor(p * (T - 1)) : started('brain') ? T - 1 : 0, k, p);
    hullMat.uniforms.uDim.value = (k === 'decoder' || k === 'output' || k === 'done') ? 1 : 0;
    inWire.uniforms.uHead.value = k === 'antenna' ? p * 1.25 : started('antenna') ? 1.25 : -1;
    outWire.uniforms.uHead.value = k === 'output' ? p * 1.25 : started('output') ? 1.25 : -1;
    antLab.alpha = k === 'antenna' || k === 'brain' ? 1 : 0.5;
    paintScreen(monScreen, [e.clean, e.decoded], ['rgba(150,160,180,.85)', '#3ddc97'], [3, 4], upto('output'), '#0d1117');

    drawSignal(PL.noisy, [e.noisy], [upto('input') < 0 ? '#d8dde5' : '#e04f24'], [1.4], upto('input') < 0 ? T - 1 : upto('input'), false);
    drawMotor(PL.motor, motorCache, upto('motor'));
    if (upto('output') < 0) drawSignal(PL.out, [e.clean], ['#d8dde5'], [1.2], T - 1, true);
    else drawSignal(PL.out, [e.clean, e.decoded], ['#9aa2b1', '#0f9d63'], [1.9, 1.4], upto('output'), true);
    if (k === 'done' && !$('score').classList.contains('show')) {
      $('score-v').textContent = `+${e.snr_gain.toFixed(1)} dB`;
      $('score').classList.add('show');
      drawBench(e.snr_gain);
    }

    if (!userCam) {
      shotT = reduce ? 1 : Math.min(1, shotT + dt / 1100);
      shotPos(shotTo, camAt);
      if (shotFrom) { const u = ease(shotT); camera.position.lerpVectors(shotFrom.pos, camAt, u); controls.target.lerpVectors(shotFrom.tgt, shotTo.at, u); }
      else { camera.position.copy(camAt); controls.target.copy(shotTo.at); }
    }
    key.position.copy(camera.position).add(new THREE.Vector3(-SPAN * 0.35, SPAN * 0.55, -SPAN * 0.2));
    fill.position.copy(camera.position).add(new THREE.Vector3(SPAN * 0.5, -SPAN * 0.1, SPAN * 0.1));
    controls.update();
    placeLabels();
    renderer.render(scene, camera);
  }
  requestAnimationFrame(frame);
  window.sim = { scene, camera, controls, SHOT, neurons, segments: keep.length };
})();
