// Fly brain EEG cleaner, staged walkthrough.
//
// One press sends a real 2 s EEGdenoiseNet epoch through the chain and the scene plays it stage by
// stage: Input -> Antenna -> Brain -> Motor -> Decoder -> Output. Nothing here is decorative:
//   - the brain hull is the MaleCNS neuropil surface, remeshed in Blender with baked ambient
//     occlusion (blender_shell.py -> hull.bin);
//   - the neurons are official MaleCNS skeletons (export_anatomy.py -> skel.bin);
//   - a neuron lights up when the synaptic wave reaches its hop distance from the antenna, scaled
//     by that neuron's recorded activity for this epoch (export_sim.py -> act_*.bin);
//   - the hull glows at the regions the wave is passing through.
(async function () {
  const DATA = './data/';
  const bin = (name, T) => fetch(DATA + name).then(r => r.arrayBuffer()).then(b => new T(b));
  const raw = name => fetch(DATA + name).then(r => r.arrayBuffer());
  const [meta, anat, skel, skelId, hullBuf, pos, group] = await Promise.all([
    fetch(DATA + 'meta.json').then(r => r.json()), fetch(DATA + 'anatomy.json').then(r => r.json()),
    bin('skel.bin', Float32Array), bin('skel_id.bin', Uint32Array), raw('hull.bin'),
    bin('pos.bin', Float32Array), bin('group.bin', Uint8Array),
  ]);
  const acts = await Promise.all(meta.epochs.map((_, k) => bin(`act_${k}.bin`, Uint8Array)));
  const OUTSIDE = new Set(['ME', 'LO', 'LOP', 'LA', 'AME', 'CV-anterior', 'CRN']);   // optic lobes and neck: not simulated
  const N = meta.n, T = meta.t, neurons = anat.neurons, S = neurons.length;
  const regions = anat.regions.map((r, k) => ({ ...r, k, out: OUTSIDE.has(r.label) }));
  const $ = id => document.getElementById(id);
  $('loading').remove();

  // ---------------------------------------------------------------- scene ----
  const view = $('view');
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
  view.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 1, 12000);

  // hull: vertices, normals, baked ambient occlusion, faces. Micrometres, skeleton frame.
  const head = new Uint32Array(hullBuf, 0, 2), HNV = head[0], HNF = head[1];
  let off = 8;
  const hullV = new Float32Array(hullBuf, off, HNV * 3); off += HNV * 12;
  const hullN = new Float32Array(hullBuf, off, HNV * 3); off += HNV * 12;
  const hullAO = new Uint8Array(hullBuf, off, HNV); off += HNV;
  const hullF = new Uint32Array(hullBuf.slice(off, off + HNF * 12));

  const c = [0, 0, 0];
  for (let i = 0; i < HNV; i++) { c[0] += hullV[3 * i]; c[1] += hullV[3 * i + 1]; c[2] += hullV[3 * i + 2]; }
  c[0] /= HNV; c[1] /= HNV; c[2] /= HNV;
  const toScene = arr => { const a = new Float32Array(arr.length); for (let i = 0; i < arr.length; i += 3) { a[i] = arr[i] - c[0]; a[i + 1] = -(arr[i + 1] - c[1]); a[i + 2] = arr[i + 2] - c[2]; } return a; };
  const flipY = arr => { const a = new Float32Array(arr.length); for (let i = 0; i < arr.length; i += 3) { a[i] = arr[i]; a[i + 1] = -arr[i + 1]; a[i + 2] = arr[i + 2]; } return a; };
  const V3 = a => new THREE.Vector3(a[0], a[1], a[2]);
  const hullPos = toScene(hullV);
  const bbox = new THREE.Box3().setFromArray(hullPos);
  const BW = bbox.max.x - bbox.min.x, BH = bbox.max.y - bbox.min.y;

  // ---- the hull: glass with baked occlusion, a cool rim, and up to 8 glowing hot spots ----
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
        float fres = pow(1.0 - abs(dot(normalize(vN), vE)), 2.4);
        float glow = 0.0;
        for (int i = 0; i < ${HOT}; i++) { float d = length(vP - uHot[i].xyz); glow += uHot[i].w * exp(-d * d / (2.0 * uR * uR)); }
        glow = clamp(glow, 0.0, 1.0);
        vec3 glass = mix(vec3(0.42, 0.49, 0.62), vec3(0.80, 0.85, 0.93), vAO);
        vec3 col = mix(glass, vec3(0.98, 0.62, 0.20), glow * 0.85);
        float a = (0.055 + 0.30 * fres + 0.35 * glow) * (1.0 - 0.55 * uDim);
        gl_FragColor = vec4(col, a);
      }`,
  });
  const hullGeom = new THREE.BufferGeometry();
  hullGeom.setAttribute('position', new THREE.BufferAttribute(hullPos, 3));
  hullGeom.setAttribute('normal', new THREE.BufferAttribute(flipY(hullN), 3));
  hullGeom.setAttribute('ao', new THREE.BufferAttribute(Float32Array.from(hullAO, v => v / 255), 1));
  hullGeom.setIndex(new THREE.BufferAttribute(hullF, 1));
  scene.add(new THREE.Mesh(hullGeom, hullMat));

  // ---- neurons: skeletons, pale at rest, their own hue when the wave reaches them ----
  const TEX_W = 4096;
  const actData = new Uint8Array(TEX_W * 4), baseData = new Uint8Array(TEX_W * 4);
  const col = new THREE.Color();
  neurons.forEach((n, s) => {
    let h = n.body >>> 0; h = Math.imul(h ^ (h >>> 16), 0x45d9f3b); h = Math.imul(h ^ (h >>> 16), 0x45d9f3b); h ^= h >>> 16;
    col.setHSL((h % 997) / 997, 0.72, 0.42);
    baseData[4 * s] = col.r * 255; baseData[4 * s + 1] = col.g * 255; baseData[4 * s + 2] = col.b * 255;
    baseData[4 * s + 3] = n.group === 1 || n.group === 2 ? 255 : 140;
  });
  const actTex = new THREE.DataTexture(actData, TEX_W, 1, THREE.RGBAFormat), baseTex = new THREE.DataTexture(baseData, TEX_W, 1, THREE.RGBAFormat);
  baseTex.needsUpdate = true;
  const lineMat = new THREE.ShaderMaterial({
    uniforms: { act: { value: actTex }, base: { value: baseTex } }, transparent: true, depthWrite: false,
    vertexShader: `attribute float nid; uniform sampler2D act; uniform sampler2D base; varying vec4 vC;
      void main() {
        vec2 uv = vec2((nid + 0.5) / ${TEX_W}.0, 0.5);
        float a = texture2D(act, uv).r; vec4 b = texture2D(base, uv);
        vec3 rest = vec3(0.74, 0.77, 0.83);
        vC = vec4(mix(rest, b.rgb, smoothstep(0.0, 0.55, a)), b.a * (0.10 + 0.85 * a));
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }`,
    fragmentShader: `varying vec4 vC; void main() { gl_FragColor = vC; }`,
  });
  const drawn = neurons.map((n, s) => n.group !== 3 || s % 4 === 0);
  const yCut = bbox.min.y - 40, zCut = bbox.max.z + 40;   // drop the neck bundle leaving the brain
  // every neuron is drawn as a sample of its branches: the whole arbor set is 1.5M segments, which
  // no laptop GPU composites at 60 fps with transparency. STRIDE thins it and keeps the shape.
  const STRIDE = 3, keep = [];
  for (let i = 0, seg = 0; i < skel.length; i += 6, seg++) {
    if (!drawn[skelId[i / 3]]) continue;
    if (seg % STRIDE) continue;
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

  // antenna anchor: centroid of the antennal nerve nodes
  const P = skelGeom.attributes.position.array, acc = new THREE.Vector3(); let an = 0;
  for (let i = 0; i < sn.length; i++) { const g = neurons[sn[i]].group; if (g === 1 || g === 2) { acc.x += P[3 * i]; acc.y += P[3 * i + 1]; acc.z += P[3 * i + 2]; an++; } }
  const ANT = acc.multiplyScalar(1 / an);

  // ---- stations: rounded plates, Input on one side, Decoder and Output on the other ----
  const INPUT = new THREE.Vector3(BW * 0.78, -BH * 0.05, 0);
  const DEC = new THREE.Vector3(-BW * 0.82, -BH * 0.12, 0);
  const OUT = new THREE.Vector3(-BW * 1.32, -BH * 0.12, 0);
  function plate(at, w, h, colour) {
    const r = 14, pts = [], seg = 7;
    const corner = (cx, cy, a0) => { for (let i = 0; i <= seg; i++) { const a = a0 + (Math.PI / 2) * (i / seg); pts.push(new THREE.Vector3(cx + r * Math.cos(a), cy + r * Math.sin(a), 0)); } };
    corner(w - r, h - r, 0); corner(-w + r, h - r, Math.PI / 2); corner(-w + r, -h + r, Math.PI); corner(w - r, -h + r, -Math.PI / 2);
    pts.push(pts[0].clone());
    const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: colour, transparent: true, opacity: 0.35 }));
    l.position.copy(at); scene.add(l); return l;
  }
  const inPlate = plate(INPUT, 46, 30, 0xe04f24), decPlate = plate(DEC, 46, 34, 0x10141c), outPlate = plate(OUT, 46, 30, 0x0f9d63);
  function screen(at, w, h) {
    const cv = document.createElement('canvas'); cv.width = 320; cv.height = 200;
    const tex = new THREE.CanvasTexture(cv);
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w * 2 - 14, h * 2 - 14),
      new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false }));
    m.position.copy(at); scene.add(m);
    return { cv, g: cv.getContext('2d'), tex };
  }
  const inScreen = screen(INPUT, 46, 30), outScreen = screen(OUT, 46, 30), decScreen = screen(DEC, 46, 34);
  function screenTrace(sc, series, colours, widths, upto) {
    const { g, cv } = sc, W = cv.width, H = cv.height;
    g.clearRect(0, 0, W, H);
    if (upto >= 0) series.forEach((sig, j) => {
      let m = 0; for (let i = 0; i < T; i++) m = Math.max(m, Math.abs(sig[i])); m = m || 1;
      g.strokeStyle = colours[j]; g.lineWidth = widths[j]; g.lineJoin = 'round'; g.beginPath();
      for (let i = 0; i <= upto; i++) { const x = i * W / (T - 1), y = H / 2 - sig[i] / m * (H / 2 - 12); i ? g.lineTo(x, y) : g.moveTo(x, y); }
      g.stroke();
    });
    sc.tex.needsUpdate = true;
  }
  (function decoderGlyph() {                        // many neurons weighted into one trace
    const { g, cv, tex } = decScreen, W = cv.width, H = cv.height;
    g.clearRect(0, 0, W, H);
    g.strokeStyle = 'rgba(16,20,28,.35)'; g.lineWidth = 2;
    for (let i = 0; i < 7; i++) { g.beginPath(); g.moveTo(14, 26 + i * (H - 52) / 6); g.lineTo(W / 2 - 6, H / 2); g.stroke(); }
    g.strokeStyle = 'rgba(15,157,99,.9)'; g.lineWidth = 3;
    g.beginPath(); g.moveTo(W / 2 - 6, H / 2); g.lineTo(W - 14, H / 2); g.stroke();
    g.fillStyle = 'rgba(16,20,28,.75)'; g.font = '600 30px system-ui, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
    g.fillText('Σ', W / 2 - 6, H / 2 - 26);
    tex.needsUpdate = true;
  })();

  // ---- connectors: a bright pulse runs along each one when its stage plays ----
  function connector(a, b, colour, bow = 0.22) {
    const mid = a.clone().lerp(b, 0.5).add(new THREE.Vector3(0, bow * a.distanceTo(b), 0));
    const pts = new THREE.QuadraticBezierCurve3(a, mid, b).getPoints(120);
    const tt = new Float32Array(pts.length);
    for (let i = 0; i < pts.length; i++) tt[i] = i / (pts.length - 1);
    const g = new THREE.BufferGeometry().setFromPoints(pts);
    g.setAttribute('t', new THREE.BufferAttribute(tt, 1));
    const m = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false,
      uniforms: { uHead: { value: -1 }, uCol: { value: new THREE.Color(colour) } },
      vertexShader: `attribute float t; uniform float uHead; varying float vA;
        void main() { float d = uHead - t; vA = (uHead < 0.0 || d < 0.0) ? 0.0 : exp(-d * d / 0.010);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: `uniform vec3 uCol; varying float vA;
        void main() { gl_FragColor = vec4(uCol, 0.12 + 0.85 * vA); }`,
    });
    scene.add(new THREE.Line(g, m));
    return m;
  }
  const inWire = connector(INPUT, ANT, 0xe04f24);
  const outWire = connector(new THREE.Vector3(DEC.x - 46, DEC.y, DEC.z), new THREE.Vector3(OUT.x + 46, OUT.y, OUT.z), 0x0f9d63, 0.18);

  // descending neurons into the decoder: a fan of thin wires, lit by the same activity texture
  const motorSlots = neurons.map((n, s) => s).filter(s => neurons[s].group === 3 && group[neurons[s].node] !== 255);
  const fan = [], fanId = [];
  for (let k = 0; k < motorSlots.length; k += 3) {
    const s = motorSlots[k]; if (!drawn[s]) continue;
    const n = neurons[s].node, p = toScene(new Float32Array([pos[3 * n], pos[3 * n + 1], pos[3 * n + 2]]));
    fan.push(p[0], p[1], p[2], DEC.x + 46, DEC.y + (Math.random() - 0.5) * 46, DEC.z + (Math.random() - 0.5) * 46); fanId.push(s, s);
  }
  const fanGeom = new THREE.BufferGeometry();
  fanGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(fan), 3));
  fanGeom.setAttribute('nid', new THREE.BufferAttribute(new Float32Array(fanId), 1));
  scene.add(new THREE.LineSegments(fanGeom, lineMat));

  // ---- labels: HTML, projected each frame ----
  const labelBox = $('labels'), labels = [];
  function label(text, at, cls = 'station', colour = '') {
    const d = document.createElement('div'); d.textContent = text; d.className = cls; if (colour) d.style.color = colour;
    labelBox.appendChild(d); const L = { d, at, alpha: 1 }; labels.push(L); return L;
  }
  label('Input', INPUT.clone().add(new THREE.Vector3(0, 48, 0)), 'station', '#e04f24');
  const antLab = label('Antenna', ANT.clone().add(new THREE.Vector3(0, -34, 0)), 'station', '#e0921b');
  const motLab = label('Motor neurons', new THREE.Vector3(DEC.x * 0.45, DEC.y + 62, 0), 'station', '#0f9d63');
  label('Decoder', DEC.clone().add(new THREE.Vector3(0, 52, 0)), 'station', '#10141c');
  label('Output', OUT.clone().add(new THREE.Vector3(0, 48, 0)), 'station', '#0f9d63');
  const regionCentre = regions.map(r => V3(toScene(new Float32Array(r.centre))));
  const flowByLabel = {};
  anat.flows.forEach(f => { const L = regions[f.dst].label; flowByLabel[L] = (flowByLabel[L] || 0) + f.w; });
  const mainLabels = Object.entries(flowByLabel).sort((a, b) => b[1] - a[1]).map(kv => kv[0]).filter(l => !OUTSIDE.has(l)).slice(0, 4);
  const regionLabel = {};
  mainLabels.forEach(l => { const r = regions.find(x => x.label === l && !x.out); if (r) regionLabel[l] = { L: label(l, regionCentre[r.k], 'region'), k: r.k }; });
  const sv = new THREE.Vector3();
  function placeLabels() {
    const w = view.clientWidth, h = view.clientHeight;
    for (const L of labels) {
      sv.copy(L.at).project(camera);
      const offscreen = sv.z > 1 || Math.abs(sv.x) > 1.05 || Math.abs(sv.y) > 1.05;
      L.d.style.left = ((sv.x + 1) / 2 * w) + 'px';
      L.d.style.top = ((1 - sv.y) / 2 * h) + 'px';
      L.d.style.opacity = offscreen ? 0 : L.alpha;
      L.d.style.visibility = (offscreen || L.alpha < 0.02) ? 'hidden' : 'visible';
    }
  }

  // ---- camera: one framing per stage, eased; dragging hands control to the viewer ----
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.08;
  const centre = new THREE.Vector3((INPUT.x + OUT.x) / 2, -BH * 0.05, 0);
  const SHOT = {
    wide: { at: centre, d: 2.30, el: 0.16, az: 0 },
    input: { at: INPUT.clone().lerp(ANT, 0.45), d: 1.15, el: 0.14, az: -0.18 },
    antenna: { at: ANT.clone().lerp(new THREE.Vector3(0, 0, 0), 0.35), d: 1.10, el: 0.10, az: -0.22 },
    brain: { at: new THREE.Vector3(0, -BH * 0.03, 0), d: 1.50, el: 0.22, az: 0.16 },
    motor: { at: new THREE.Vector3(DEC.x * 0.42, DEC.y + 20, 0), d: 1.60, el: 0.18, az: -0.10 },
    decoder: { at: DEC.clone().lerp(new THREE.Vector3(0, 0, 0), 0.30), d: 1.40, el: 0.12, az: -0.05 },
    output: { at: DEC.clone().lerp(OUT, 0.55), d: 1.30, el: 0.10, az: 0 },
  };
  const SPAN = INPUT.distanceTo(OUT);
  const camAt = new THREE.Vector3();
  const shotPos = (s, out) => { const d = s.d * SPAN * 0.58; return out.set(s.at.x + Math.sin(s.az) * d, s.at.y + Math.sin(s.el) * d, s.at.z - Math.cos(s.az) * Math.cos(s.el) * d); };
  let userCam = false, shotFrom = null, shotTo = SHOT.wide, shotT = 1;
  function setShot(s) { if (!s || s === shotTo) return; shotFrom = { pos: camera.position.clone(), tgt: controls.target.clone() }; shotTo = s; shotT = 0; }
  shotPos(SHOT.wide, camera.position); controls.target.copy(SHOT.wide.at);

  function resize() {
    const w = view.clientWidth, h = view.clientHeight;
    renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
    const dpr = Math.min(devicePixelRatio, 2);
    for (const id of ['noisy', 'motor', 'out']) {
      const cv = $(id), ch = id === 'motor' ? 96 : 72;
      cv.style.height = ch + 'px'; cv.width = (cv.clientWidth || 340) * dpr; cv.height = ch * dpr;
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
  const HOPS = 4;                                    // synaptic layers the wave sweeps through
  const dev = new Float32Array(S), regAct = new Float32Array(regions.length);
  function activity(sample, key, p) {
    const a = acts[epoch], { mu, sd } = stats[epoch], o = sample * N;
    for (let s = 0; s < S; s++) {
      const n = neurons[s], ant = n.group === 1 || n.group === 2;
      const rec = ant ? a[o + n.node] / 255 : Math.min(1, Math.max(0, (a[o + n.node] - mu[s]) / (2.5 * sd[s] + 3)));
      const hop = n.hop < 0 ? HOPS : Math.min(n.hop, HOPS);
      let v = (key === 'idle' || key === 'input') ? 0.30 : 0;   // the specimen stays visible before the wave
      if (key === 'antenna') v = ant ? Math.min(1, p * 1.4) : 0;
      else if (key === 'brain') { const w = Math.min(1, Math.max(0, (p * (HOPS + 0.9) - hop) / 0.55)); v = ant ? 0.75 + 0.25 * rec : w * (0.35 + 0.65 * rec); }
      else if (key === 'motor') v = n.group === 3 ? 0.65 + 0.35 * Math.sin(Math.PI * Math.min(1, p)) : 0.26 * (0.4 + 0.6 * rec);
      else if (key === 'decoder' || key === 'output' || key === 'done') v = n.group === 3 ? 0.8 : 0.20;
      dev[s] = v; actData[4 * s] = v * 255;
    }
    actTex.needsUpdate = true;
    for (let k = 0; k < regions.length; k++) {
      if (regions[k].out || key !== 'brain') { regAct[k] = 0; continue; }
      let num = 0, den = 0;
      for (const [s, w] of regW[k]) { num += w * dev[s]; den += w; }
      regAct[k] = den ? Math.min(1, 2.4 * num / den) : 0;
    }
    const top = regions.filter(r => !r.out).sort((x, y) => regAct[y.k] - regAct[x.k]).slice(0, HOT);
    for (let i = 0; i < HOT; i++) { const r = top[i]; hot[i].set(regionCentre[r.k].x, regionCentre[r.k].y, regionCentre[r.k].z, regAct[r.k]); }
    for (const l in regionLabel) {
      const { L, k } = regionLabel[l];
      L.alpha = regAct[k] > 0.28 ? Math.min(1, (regAct[k] - 0.28) * 4) : 0;
      L.d.style.color = '#c8401f';
    }
  }

  // ----------------------------------------------------------------- plots ----
  const PL = { noisy: $('noisy'), motor: $('motor'), out: $('out') };
  const motorShow = motorSlots.slice().sort((a, b) => stats[0].sd[b] - stats[0].sd[a]).slice(0, 6);
  const grid = (g, W, H) => { g.strokeStyle = '#f0f3f8'; g.lineWidth = 1; for (let i = 1; i < 4; i++) { const x = i * W / 4; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H); g.stroke(); } };
  const cursor = (g, W, H, upto) => { if (upto >= T - 1) return; const x = upto * W / (T - 1); g.strokeStyle = 'rgba(47,109,240,.5)'; g.lineWidth = 1; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H); g.stroke(); };
  const amp = series => { let m = 0; for (const s of series) for (let i = 0; i < T; i++) m = Math.max(m, Math.abs(s[i])); return m || 1; };
  function drawSignal(cv, series, colours, widths, upto, fill, shared) {
    const g = cv.getContext('2d'), W = cv.clientWidth, H = parseFloat(cv.style.height);
    g.clearRect(0, 0, W, H); grid(g, W, H);
    if (upto < 0) return;
    const common = shared ? amp(series) : 0;
    series.forEach((s, j) => {
      const m = common || amp([s]);
      const X = i => i * W / (T - 1), Y = i => H / 2 - s[i] / m * (H / 2 - 5);
      if (fill && j === 0) {
        const grad = g.createLinearGradient(0, 0, 0, H);
        grad.addColorStop(0, 'rgba(224,79,36,.15)'); grad.addColorStop(1, 'rgba(224,79,36,0)');
        g.fillStyle = grad; g.beginPath(); g.moveTo(0, H / 2);
        for (let i = 0; i <= upto; i++) g.lineTo(X(i), Y(i));
        g.lineTo(X(upto), H / 2); g.closePath(); g.fill();
      }
      g.strokeStyle = colours[j]; g.lineWidth = widths[j]; g.lineJoin = 'round'; g.beginPath();
      for (let i = 0; i <= upto; i++) { const x = X(i), y = Y(i); i ? g.lineTo(x, y) : g.moveTo(x, y); }
      g.stroke();
    });
    cursor(g, W, H, upto);
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
    cursor(g, W, H, upto);
  }
  // where this epoch's result sits against the baselines measured on 500 held-out epochs.
  // The shuffled-wiring control matches the real connectome, so it is on the scale, not hidden.
  const BENCH = [
    { v: 0, name: 'no processing', tone: '#b9c0cc' },
    { v: 8.1, name: 'linear filter, no brain', tone: '#8e97a6' },
    { v: 10.4, name: 'shuffled wiring', tone: '#c8401f' },
    { v: 10.4, name: 'this connectome', tone: '#0f9d63' },
  ];
  function drawBench(value) {
    const cv = $('bench'), g = cv.getContext('2d'), W = cv.clientWidth, H = 62;
    const dpr = Math.min(devicePixelRatio, 2);
    if (cv.width !== W * dpr) { cv.width = W * dpr; cv.height = H * dpr; }
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const L = 8, R = W - 8, lo = -0.6, hi = 13, X = v => L + (v - lo) / (hi - lo) * (R - L), y = 34;
    g.strokeStyle = '#e3e7ee'; g.lineWidth = 2; g.beginPath(); g.moveTo(L, y); g.lineTo(R, y); g.stroke();
    g.font = '500 9.5px Inter, system-ui, sans-serif'; g.textBaseline = 'middle';
    BENCH.forEach((b, i) => {
      const x = X(b.v), up = i === 2;                       // shuffled and connectome coincide: stack them
      g.strokeStyle = b.tone; g.lineWidth = 2;
      g.beginPath(); g.moveTo(x, y - (up ? 9 : 0)); g.lineTo(x, y + (up ? 0 : 9)); g.stroke();
      g.fillStyle = b.tone; g.textAlign = i === 0 ? 'left' : 'center';
      g.fillText(b.name, i === 0 ? L : Math.min(x, R - 46), up ? y - 15 : y + 17);
    });
    g.fillStyle = '#10141c'; g.textAlign = 'center';
    const x = X(value);
    g.beginPath(); g.arc(x, y, 4.5, 0, 7); g.fill();
    g.font = '600 10.5px Inter, system-ui, sans-serif';
    g.fillText(`${value.toFixed(1)} dB`, Math.min(Math.max(x, L + 20), R - 20), y - 26);
    g.fillStyle = '#9aa2b1'; g.font = '500 9.5px Inter, system-ui, sans-serif'; g.textAlign = 'right';
    g.fillText('SNR gain, mean of 500 held-out epochs', R, H - 4);
  }

  let motorCache = null, motorEpoch = -1;
  const motorTrace = () => motorShow.map(s => { const a = acts[epoch], n = neurons[s].node, out = new Float32Array(T); for (let i = 0; i < T; i++) out[i] = a[i * N + n]; return out; });

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
  STAGE.forEach((s, j) => {
    const d = document.createElement('button'); d.className = 'step'; d.type = 'button';
    d.innerHTML = `<div class="bar"><i></i></div><span>${s.name}</span>`;
    d.title = `jump to ${s.name}`;
    d.onclick = () => { elapsed = START[j]; playing = false; idx = -2; syncButtons(); };   // jump and hold
    rail.appendChild(d); s.el = d; s.fill = d.querySelector('i');
  });
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
  $('pause').onclick = () => { playing ? playing = false : play(elapsed); syncButtons(); };
  $('camreset').onclick = () => { userCam = false; $('camchip').classList.remove('show'); };
  controls.addEventListener('start', () => { userCam = true; $('camchip').classList.add('show'); });
  const epochBox = $('epochs'), names = ['Hard', 'Typical', 'Mild'];
  meta.epochs.forEach((e, k) => {
    const b = document.createElement('button');
    b.textContent = names[k] || 'Epoch ' + k;
    b.onclick = () => setEpoch(k);
    epochBox.appendChild(b);
  });
  const LIVE = { input: 'n-in', antenna: 'n-ant', brain: 'n-brain', motor: 'n-motor', decoder: 'n-dec', output: 'n-out' };
  function setEpoch(k) {
    epoch = k; elapsed = -1; playing = false; idx = -2;
    [...epochBox.children].forEach((b, i) => { b.classList.toggle('on', i === k); b.setAttribute('aria-checked', i === k); });
    $('score').classList.remove('show');
    $('send').querySelector('b').textContent = 'Send epoch'; $('send').querySelector('i').style.width = '0';
    syncButtons();
    STAGE.forEach(s => { s.el.classList.remove('on', 'done'); s.fill.style.width = '0'; });
    $('rail').classList.remove('finished');
    $('in-snr').textContent = 'noisy EEG'; $('out-cc').textContent = 'grey = true EEG';
    ['n-in', 'n-mot', 'n-out', 'n-ant', 'n-brain', 'n-motor', 'n-dec'].forEach(id => $(id).classList.remove('live'));
    motorEpoch = -1;
    userCam = false; setShot(SHOT.wide);
  }
  $('send').onclick = () => {
    play(0); idx = -2; userCam = false;
    $('camchip').classList.remove('show');
    $('score').classList.remove('show');
    syncButtons();
  };
  setEpoch(1);

  const ease = u => u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
  let last = performance.now();
  function frame(now) {
    requestAnimationFrame(frame);
    const dt = Math.min(now - last, 100); last = now;
    const e = meta.epochs[epoch];
    if (motorEpoch !== epoch) { motorCache = motorTrace(); motorEpoch = epoch; }

    if (playing) { elapsed = now - origin; if (elapsed >= TOTAL) { elapsed = TOTAL; playing = false; syncButtons(); } }
    let key = 'idle', p = 0, i = -1;
    if (elapsed >= 0) {
      let el = elapsed; i = 0;
      while (i < STAGE.length && el > STAGE[i].dur) { el -= STAGE[i].dur; i++; }
      if (i >= STAGE.length) { key = 'done'; i = STAGE.length - 1; p = 1; }
      else { key = STAGE[i].key; p = el / STAGE[i].dur; }
    }
    const started = k => elapsed >= 0 && STAGE.findIndex(s => s.key === k) <= i;
    const upto = k => key === k ? Math.floor(p * (T - 1)) : started(k) ? T - 1 : -1;

    if (i !== idx) {
      idx = i;
      STAGE.forEach((s, j) => { s.el.classList.toggle('on', j === i && key !== 'done'); s.el.classList.toggle('done', j < i || key === 'done'); });
      rail.classList.toggle('finished', key === 'done');
      $('in-snr').textContent = started('input') ? `${e.snr_in.toFixed(1)} dB in` : 'noisy EEG';
      $('out-cc').textContent = started('output') ? `r = ${e.cc.toFixed(2)} with true EEG` : 'grey = true EEG';
      Object.values(LIVE).forEach(id => $(id).classList.remove('live'));
      $('n-mot').classList.remove('live'); $('n-in').classList.remove('live'); $('n-out').classList.remove('live');
      if (LIVE[key]) $(LIVE[key]).classList.add('live');
      if (key === 'motor') $('n-mot').classList.add('live');
      if (!userCam) setShot(SHOT[key === 'done' || key === 'idle' ? 'wide' : STAGE[i].shot]);
      const lab = $('send').querySelector('b');
      if (key === 'idle') lab.textContent = 'Send epoch';
      else if (key === 'done') lab.textContent = 'Send again';
      else lab.textContent = `${STAGE[i].name} · ${i + 1} of ${STAGE.length}`;
      $('live').textContent = key === 'done' ? `Finished. ${meta.epochs[epoch].snr_gain.toFixed(1)} decibels cleaner.`
        : key === 'idle' ? '' : `Stage ${i + 1} of ${STAGE.length}, ${STAGE[i].name}`;
      syncButtons();
    }
    $('send').querySelector('i').style.width = elapsed < 0 ? '0' : (100 * Math.min(1, elapsed / TOTAL)) + '%';
    if (i >= 0 && key !== 'done') STAGE[i].fill.style.width = (p * 100) + '%';

    const brainT = key === 'brain' ? Math.floor(p * (T - 1)) : started('brain') ? T - 1 : 0;
    activity(brainT, key, p);
    hullMat.uniforms.uDim.value = (key === 'decoder' || key === 'output' || key === 'done') ? 1 : 0;
    inWire.uniforms.uHead.value = key === 'antenna' ? p * 1.25 : started('antenna') ? 1.25 : -1;
    outWire.uniforms.uHead.value = key === 'output' ? p * 1.25 : started('output') ? 1.25 : -1;
    inPlate.material.opacity = key === 'input' || key === 'antenna' ? 1 : 0.35;
    decPlate.material.opacity = key === 'decoder' || key === 'output' || key === 'done' ? 1 : 0.35;
    outPlate.material.opacity = key === 'output' || key === 'done' ? 1 : 0.35;
    antLab.alpha = key === 'antenna' || key === 'brain' ? 1 : 0.55;
    motLab.alpha = key === 'motor' || key === 'decoder' ? 1 : 0.5;
    antLab.d.classList.toggle('hot', key === 'antenna' || key === 'brain');
    motLab.d.classList.toggle('hot', key === 'motor');

    screenTrace(inScreen, [e.noisy], ['#e04f24'], [3], upto('input'));
    screenTrace(outScreen, [e.clean, e.decoded], ['rgba(140,148,164,.8)', '#0f9d63'], [2, 3.4], upto('output'));
    if (upto('input') < 0) drawSignal(PL.noisy, [e.noisy], ['#d6dbe4'], [1.2], T - 1, false);
    else drawSignal(PL.noisy, [e.noisy], ['#e04f24'], [1.5], upto('input'), true);
    drawMotor(PL.motor, motorCache, upto('motor'));
    if (upto('output') < 0) drawSignal(PL.out, [e.clean], ['#d6dbe4'], [1.2], T - 1, false, true);
    else drawSignal(PL.out, [e.clean, e.decoded], ['#9aa2b1', '#0f9d63'], [1.9, 1.5], upto('output'), false, true);
    if (key === 'done' && !$('score').classList.contains('show')) {
      $('score-v').textContent = `+${e.snr_gain.toFixed(1)} dB`;
      $('score-t').innerHTML = `cleaner than the input, this epoch<br>scrambling the wiring scores the same: the size and dynamics do the work`;
      $('score').classList.add('show');
      drawBench(e.snr_gain);
    }

    if (!userCam) {
      shotT = reduce ? 1 : Math.min(1, shotT + dt / 1100);
      shotPos(shotTo, camAt);
      if (shotFrom) { const u = ease(shotT); camera.position.lerpVectors(shotFrom.pos, camAt, u); controls.target.lerpVectors(shotFrom.tgt, shotTo.at, u); }
      else { camera.position.copy(camAt); controls.target.copy(shotTo.at); }
    }
    controls.update();
    placeLabels();
    renderer.render(scene, camera);
  }
  requestAnimationFrame(frame);
  window.sim = { scene, camera, controls, SHOT, regions, neurons };
})();
