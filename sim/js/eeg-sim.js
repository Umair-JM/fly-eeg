// Fly brain EEG cleaner. One button sends a real 2 s EEG epoch through the chain and the scene plays
// it stage by stage: Input -> Antenna -> Brain (recorded activity of every neuron, regions glowing in
// the order the synapse table gives) -> Motor neurons -> Decoder -> Output. Anatomy: official MaleCNS
// skeletons and neuropil meshes (export_anatomy.py); activity: export_sim.py.
(async function () {
  const DATA = './data/';
  const bin = (name, T) => fetch(DATA + name).then(r => r.arrayBuffer()).then(b => new T(b));
  const [meta, anat, skel, skelId, shellBuf, pos, group] = await Promise.all([
    fetch(DATA + 'meta.json').then(r => r.json()), fetch(DATA + 'anatomy.json').then(r => r.json()),
    bin('skel.bin', Float32Array), bin('skel_id.bin', Uint32Array), fetch(DATA + 'shell.bin').then(r => r.arrayBuffer()),
    bin('pos.bin', Float32Array), bin('group.bin', Uint8Array),
  ]);
  const acts = await Promise.all(meta.epochs.map((_, k) => bin(`act_${k}.bin`, Uint8Array)));
  const OUTSIDE = new Set(['ME', 'LO', 'LOP', 'LA', 'AME', 'CV-anterior', 'CRN']);   // optic lobes and neck: not simulated
  const N = meta.n, T = meta.t, neurons = anat.neurons, S = neurons.length;
  const regions = anat.regions.map((r, k) => ({ ...r, k, out: OUTSIDE.has(r.label) }));
  document.getElementById('loading').remove();

  // ---- coordinates: micrometres, centred on the brain, EM y flipped so dorsal is up ----
  const hdr = new Uint32Array(shellBuf, 0, 2), NV = hdr[0], NF = hdr[1];
  const shellV = new Float32Array(shellBuf, 8, NV * 3), shellF = new Uint32Array(shellBuf, 8 + NV * 12, NF * 3);
  const c = [0, 0, 0]; let nc = 0, yMax = -1e9, zMax = -1e9;
  for (const r of regions) { if (r.out) continue; for (let i = r.v0; i < r.v0 + r.nv; i++) { c[0] += shellV[3 * i]; c[1] += shellV[3 * i + 1]; c[2] += shellV[3 * i + 2]; nc++; yMax = Math.max(yMax, shellV[3 * i + 1]); zMax = Math.max(zMax, shellV[3 * i + 2]); } }
  c[0] /= nc; c[1] /= nc; c[2] /= nc;
  const toScene = arr => { const o = new Float32Array(arr.length); for (let i = 0; i < arr.length; i += 3) { o[i] = arr[i] - c[0]; o[i + 1] = -(arr[i + 1] - c[1]); o[i + 2] = arr[i + 2] - c[2]; } return o; };
  const V3 = a => new THREE.Vector3(a[0], a[1], a[2]);

  const view = document.getElementById('view');
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  view.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xffffff);
  const camera = new THREE.PerspectiveCamera(40, 1, 1, 6000);
  scene.add(new THREE.AmbientLight(0xffffff, 0.7));
  const key = new THREE.DirectionalLight(0xffffff, 0.6); scene.add(key);

  // ---- neuropil shell: one glassy mesh per region ----
  const shellPos = new THREE.BufferAttribute(toScene(shellV), 3);
  const regionMesh = regions.map(r => {
    if (r.out) return null;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', shellPos);
    g.setIndex(new THREE.BufferAttribute(shellF.subarray(r.f0 * 3, (r.f0 + r.nf) * 3), 1));
    g.computeVertexNormals();
    const m = new THREE.Mesh(g, new THREE.MeshPhongMaterial({ color: 0x8a97b2, transparent: true, opacity: 0.09, depthWrite: false, side: THREE.DoubleSide, shininess: 60 }));
    scene.add(m);
    return m;
  });
  const regionCentre = regions.map(r => V3(toScene(new Float32Array(r.centre))));

  // ---- neurons: official skeletons; colour per neuron from two 1-D textures (base colour, activity) ----
  const TEX_W = 4096;
  const actData = new Uint8Array(TEX_W * 4), baseData = new Uint8Array(TEX_W * 4);
  const col = new THREE.Color();
  neurons.forEach((n, s) => {
    let h = n.body >>> 0; h = Math.imul(h ^ (h >>> 16), 0x45d9f3b); h = Math.imul(h ^ (h >>> 16), 0x45d9f3b); h ^= h >>> 16;
    col.setHSL((h % 360) / 360, 0.85, 0.45);
    baseData[4 * s] = col.r * 255; baseData[4 * s + 1] = col.g * 255; baseData[4 * s + 2] = col.b * 255;
    baseData[4 * s + 3] = n.group === 1 || n.group === 2 ? 255 : 120;
  });
  const actTex = new THREE.DataTexture(actData, TEX_W, 1, THREE.RGBAFormat), baseTex = new THREE.DataTexture(baseData, TEX_W, 1, THREE.RGBAFormat);
  baseTex.needsUpdate = true;
  const lineMat = new THREE.ShaderMaterial({
    uniforms: { act: { value: actTex }, base: { value: baseTex } }, transparent: true, depthWrite: false,
    vertexShader: `attribute float nid; uniform sampler2D act; uniform sampler2D base; varying vec4 vC;
      void main() { vec2 uv = vec2((nid + 0.5) / ${TEX_W}.0, 0.5); float a = texture2D(act, uv).r; vec4 b = texture2D(base, uv);
        vC = vec4(b.rgb * (0.95 - 0.35 * a), b.a * (0.20 + 0.75 * a)); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: `varying vec4 vC; void main() { gl_FragColor = vC; }`,
  });
  const drawn = neurons.map((n, s) => n.group !== 3 || s % 4 === 0);     // a quarter of the descending neurons is enough to see
  const keepSeg = [], yCut = yMax + 30, zCut = zMax + 30;                 // beyond the brain shell = neck and body
  for (let i = 0; i < skel.length; i += 6) if (drawn[skelId[i / 3]] && skel[i + 1] < yCut && skel[i + 4] < yCut && skel[i + 2] < zCut && skel[i + 5] < zCut) keepSeg.push(i / 6);
  const skelPos = new Float32Array(keepSeg.length * 6), skelNid = new Float32Array(keepSeg.length * 2);
  keepSeg.forEach((sg, j) => { for (let q = 0; q < 6; q++) skelPos[6 * j + q] = skel[6 * sg + q]; skelNid[2 * j] = skelId[2 * sg]; skelNid[2 * j + 1] = skelId[2 * sg + 1]; });
  const skelGeom = new THREE.BufferGeometry();
  skelGeom.setAttribute('position', new THREE.BufferAttribute(toScene(skelPos), 3));
  skelGeom.setAttribute('nid', new THREE.BufferAttribute(skelNid, 1));
  scene.add(new THREE.LineSegments(skelGeom, lineMat));

  // where the antennal nerve enters: the outermost point of the antenna neurons' skeletons
  const P = skelGeom.attributes.position.array; let far = [0, 0, 0], fd = -1;
  for (let i = 0; i < skelNid.length; i++) {
    const g = neurons[skelNid[i]].group; if (g !== 1 && g !== 2) continue;
    const d = P[3 * i] ** 2 + P[3 * i + 1] ** 2 + P[3 * i + 2] ** 2;
    if (d > fd) { fd = d; far = [P[3 * i], P[3 * i + 1], P[3 * i + 2]]; }
  }
  const ANT = V3(far), INPUT = ANT.clone().multiplyScalar(2.1);

  // ---- blocks: input, decoder, output; wires from a sample of descending neurons into the decoder ----
  const DEC = new THREE.Vector3(-400, -40, 0), OUT = new THREE.Vector3(DEC.x - 190, DEC.y, DEC.z);
  function block(at, color) {
    const b = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(56, 56, 56)), new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.55 }));
    b.position.copy(at); scene.add(b); return b;
  }
  const inBlock = block(INPUT, 0xe8562a), decBlock = block(DEC, 0x178f60), outBlock = block(OUT, 0x178f60);
  const link = (a, b, color) => { const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints([a, b]), new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.35 })); scene.add(l); return l; };
  const inLink = link(INPUT, ANT, 0xe8562a), outLink = link(new THREE.Vector3(DEC.x - 28, DEC.y, DEC.z), new THREE.Vector3(OUT.x + 28, OUT.y, OUT.z), 0x178f60);
  const motorSlots = neurons.map((n, s) => s).filter(s => neurons[s].group === 3 && group[neurons[s].node] !== 255);
  const wires = [], wireId = [];
  for (let k = 0; k < motorSlots.length; k += 4) {
    const s = motorSlots[k]; if (!drawn[s]) continue;
    const n = neurons[s].node, p = toScene(new Float32Array([pos[3 * n], pos[3 * n + 1], pos[3 * n + 2]]));
    wires.push(p[0], p[1], p[2], DEC.x + 28, DEC.y + (Math.random() - 0.5) * 44, DEC.z + (Math.random() - 0.5) * 44); wireId.push(s, s);
  }
  const wireGeom = new THREE.BufferGeometry();
  wireGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(wires), 3));
  wireGeom.setAttribute('nid', new THREE.BufferAttribute(new Float32Array(wireId), 1));
  scene.add(new THREE.LineSegments(wireGeom, lineMat));

  // ---- the route as pulses: one arc per synapse flow (hop k -> k+1), then DN regions -> decoder -> output ----
  const flows = anat.flows.map(f => ({ a: f.src < 0 ? ANT : regionCentre[f.src], b: regionCentre[f.dst], hop: f.hop + 1, w: f.w, src: f.src, dst: f.dst }));
  const maxHop = Math.max(...flows.map(f => f.hop));
  flows.unshift({ a: INPUT, b: ANT, hop: 0, w: Math.max(...flows.map(f => f.w)), src: -2, dst: -1 });
  for (const [rid, n] of anat.dn_regions.slice(0, 3)) flows.push({ a: regionCentre[rid], b: DEC, hop: maxHop + 1, w: n * 80, src: rid, dst: -3 });
  flows.push({ a: DEC, b: OUT, hop: maxHop + 2, w: 4000, src: -3, dst: -4 });
  const LAST = maxHop + 2, wMax = Math.max(...flows.map(f => f.w));
  const HOP_COL = ['#e8562a', '#e8562a', '#d9412f', '#b8336a', '#7d3fa0', '#178f60', '#178f60', '#178f60'];
  const parts = [];
  flows.forEach(f => {
    const n = Math.round(8 + 50 * Math.sqrt(f.w / wMax));
    const mid = f.a.clone().add(f.b).multiplyScalar(0.5), dir = f.b.clone().sub(f.a), len = dir.length();
    const ctrl = mid.add(new THREE.Vector3(-dir.y, dir.x, 0).normalize().multiplyScalar(0.22 * len)).add(new THREE.Vector3(0, 0.18 * len, 0));
    const cc = new THREE.Color(HOP_COL[Math.min(f.hop, HOP_COL.length - 1)]);
    for (let i = 0; i < n; i++) parts.push({ f, ctrl, off: Math.random() * 0.45, cc, jitter: new THREE.Vector3((Math.random() - 0.5) * 20, (Math.random() - 0.5) * 20, (Math.random() - 0.5) * 20) });
  });
  const NP = parts.length, pPos = new Float32Array(NP * 3), pCol = new Float32Array(NP * 3), pSize = new Float32Array(NP), pAlpha = new Float32Array(NP);
  const pGeom = new THREE.BufferGeometry();
  pGeom.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
  pGeom.setAttribute('color', new THREE.BufferAttribute(pCol, 3));
  pGeom.setAttribute('size', new THREE.BufferAttribute(pSize, 1));
  pGeom.setAttribute('alpha', new THREE.BufferAttribute(pAlpha, 1));
  scene.add(new THREE.Points(pGeom, new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, depthTest: false,
    vertexShader: `attribute float size; attribute float alpha; attribute vec3 color; varying vec4 vC;
      void main() { vC = vec4(color, alpha); vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = size * (900.0 / -mv.z) * ${Math.min(devicePixelRatio, 2).toFixed(1)}; gl_Position = projectionMatrix * mv; }`,
    fragmentShader: `varying vec4 vC; void main() { float d = length(gl_PointCoord - 0.5); if (d > 0.5) discard;
      gl_FragColor = vec4(vC.rgb, vC.a * smoothstep(0.5, 0.15, d)); }`,
  })));
  const tmp = new THREE.Vector3();
  // hopProgress[h] in [0, 1+]: how far the pulse of hop h has travelled; each particle starts at its own offset
  function placeParticles(hopProgress) {
    for (let i = 0; i < NP; i++) {
      const q = parts[i], u = Math.min(1, Math.max(0, (hopProgress[q.f.hop] - q.off) / (1 - 0.45)));
      const on = hopProgress[q.f.hop] > q.off && u < 1;
      const a = q.f.a, b = q.f.b, k = q.ctrl;
      tmp.set((1 - u) * (1 - u) * a.x + 2 * (1 - u) * u * k.x + u * u * b.x, (1 - u) * (1 - u) * a.y + 2 * (1 - u) * u * k.y + u * u * b.y, (1 - u) * (1 - u) * a.z + 2 * (1 - u) * u * k.z + u * u * b.z).add(q.jitter);
      pPos[3 * i] = tmp.x; pPos[3 * i + 1] = tmp.y; pPos[3 * i + 2] = tmp.z;
      pCol[3 * i] = q.cc.r; pCol[3 * i + 1] = q.cc.g; pCol[3 * i + 2] = q.cc.b;
      pSize[i] = on ? 7 : 0; pAlpha[i] = on ? 0.9 * Math.sin(Math.PI * u) + 0.1 : 0;
    }
    pGeom.attributes.position.needsUpdate = pGeom.attributes.color.needsUpdate = pGeom.attributes.size.needsUpdate = pGeom.attributes.alpha.needsUpdate = true;
  }

  // ---- labels: HTML, projected every frame, so they stay crisp ----
  const labelBox = document.getElementById('labels'), labels = [];
  function label(text, at, cls = '', color = '') {
    const d = document.createElement('div'); d.textContent = text; d.className = cls; if (color) d.style.color = color;
    labelBox.appendChild(d); const L = { d, at, alpha: 1 }; labels.push(L); return L;
  }
  label('Input', INPUT.clone().add(new THREE.Vector3(0, 48, 0)), 'big', '#e8562a');
  label('Antenna', ANT.clone().add(new THREE.Vector3(0, -26, 0)), 'big', '#e08a1e');
  label('Motor', DEC.clone().add(new THREE.Vector3(150, 40, 0)), 'big', '#178f60');
  label('Decoder', DEC.clone().add(new THREE.Vector3(0, 50, 0)), 'big', '#178f60');
  label('Output', OUT.clone().add(new THREE.Vector3(0, 50, 0)), 'big', '#178f60');
  // name only the main stations: the five region labels that carry the most synapse flow
  const flowByLabel = {};
  flows.forEach(f => { if (f.dst >= 0) flowByLabel[regions[f.dst].label] = (flowByLabel[regions[f.dst].label] || 0) + f.w; });
  const mainLabels = new Set(Object.entries(flowByLabel).sort((a, b) => b[1] - a[1]).slice(0, 5).map(kv => kv[0]));
  const regionLabel = {};
  regions.forEach(r => { if (!r.out && mainLabels.has(r.label) && !regionLabel[r.label]) regionLabel[r.label] = label(r.label, regionCentre[r.k], 'region'); });
  const sv = new THREE.Vector3();
  function placeLabels() {
    const w = view.clientWidth, h = view.clientHeight;
    for (const L of labels) {
      sv.copy(L.at).project(camera);
      const hidden = sv.z > 1 || sv.x < -1.1 || sv.x > 1.1 || sv.y < -1.1 || sv.y > 1.1;
      L.d.style.left = ((sv.x + 1) / 2 * w) + 'px'; L.d.style.top = ((1 - sv.y) / 2 * h) + 'px';
      L.d.style.opacity = hidden ? 0 : L.alpha;
    }
  }

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.autoRotate = false; controls.enableDamping = true;
  controls.target.set(-230, -30, 0);
  camera.position.set(-230, 60, -1200);
  function resize() { const w = view.clientWidth, h = view.clientHeight; renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix(); }
  addEventListener('resize', resize); resize();

  // ---- per-neuron and per-region activity ----
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
  const dev = new Float32Array(S), regAct = new Float32Array(regions.length);
  // gate: 0 = brain at rest (before the signal arrives), 1 = playing the recorded activity
  function activity(sample, gate, motorGlow) {
    const a = acts[epoch], { mu, sd } = stats[epoch], off = sample * N;
    for (let s = 0; s < S; s++) {
      const n = neurons[s], raw = a[off + n.node];
      let v = n.group === 1 || n.group === 2 ? raw / 255 : Math.min(1, Math.max(0, (raw - mu[s]) / (2.5 * sd[s] + 3)));
      v *= gate;
      if (n.group === 3) v = Math.max(v, motorGlow);
      dev[s] = v; actData[4 * s] = v * 255;
    }
    actTex.needsUpdate = true;
    regions.forEach((r, k) => {
      if (r.out) return;
      let num = 0, den = 0;
      for (const [s, w] of regW[k]) { num += w * dev[s]; den += w; }
      const v = den ? Math.min(1, 2.2 * num / den) : 0;
      regAct[k] = v;
      const m = regionMesh[k].material;
      m.opacity = 0.09 + 0.35 * v; m.color.setRGB(0.54 + 0.4 * v, 0.59 - 0.2 * v, 0.70 - 0.55 * v);
    });
    for (const lab in regionLabel) {
      let v = 0; regions.forEach(r => { if (r.label === lab) v = Math.max(v, regAct[r.k]); });
      regionLabel[lab].alpha = 0.5 + 0.5 * v; regionLabel[lab].d.style.color = v > 0.3 ? '#c8401f' : '#5b6478';
    }
  }

  // ---- panel ----
  const $ = id => document.getElementById(id);
  const plots = { noisy: $('noisy'), motor: $('motor'), out: $('out') };
  for (const k in plots) { plots[k].width = 800; plots[k].height = k === 'motor' ? 300 : 220; }
  const motorShow = motorSlots.slice().sort((a, b) => stats[0].sd[b] - stats[0].sd[a]).slice(0, 6);
  let epoch = 1, stageT = null, stageIdx = 0;
  const epochBox = $('epochs'), names = ['hard', 'typical', 'mild'];
  meta.epochs.forEach((e, k) => { const b = document.createElement('button'); b.textContent = names[k] || 'epoch ' + k; b.onclick = () => setEpoch(k); epochBox.appendChild(b); });
  function setEpoch(k) {
    epoch = k; stageT = null;
    [...epochBox.children].forEach((b, i) => b.classList.toggle('on', i === k));
    $('gain').textContent = '';
  }
  setEpoch(1);

  // ---- the staged run: Input -> Antenna -> Brain -> Motor -> Decoder -> Output ----
  const STAGE = [
    { name: 'input', dur: 1500 },      // the epoch is drawn in the input plot
    { name: 'antenna', dur: 1300 },    // pulse from the input block to the antenna
    { name: 'brain', dur: 4500 },      // recorded activity plays through the epoch; pulses hop through the regions
    { name: 'motor', dur: 1500 },      // descending / motor neurons light up, motor traces drawn
    { name: 'decoder', dur: 1300 },    // pulse from the motor regions into the decoder
    { name: 'output', dur: 1300 },     // pulse from the decoder to the output block; final output drawn
    { name: 'done', dur: 1e12 },
  ];
  $('send').onclick = () => { stageT = performance.now(); stageIdx = 0; $('gain').textContent = ''; };
  const hopProgress = new Float32Array(LAST + 1);

  function trace(cv, series, colors, upto, lineW, rows = 1) {
    const g = cv.getContext('2d'), W = cv.width, H = cv.height, h = H / rows;
    g.clearRect(0, 0, W, H);
    if (upto < 0) return;
    series.forEach((s, j) => {
      let m = 0; for (let i = 0; i < T; i++) m = Math.max(m, Math.abs(s[i]));
      if (rows > 1) { let mu = 0; for (let i = 0; i < T; i++) mu += s[i]; mu /= T; m = 0; for (let i = 0; i < T; i++) m = Math.max(m, Math.abs(s[i] - mu)); s = s.map(v => v - mu); }
      const y0 = rows > 1 ? h * (j + 0.5) : H / 2, amp = rows > 1 ? h / 2 - 3 : H / 2 - 6;
      g.strokeStyle = colors[j % colors.length]; g.lineWidth = lineW[j % lineW.length]; g.beginPath();
      for (let i = 0; i <= upto; i++) { const x = i * W / (T - 1), y = y0 - s[i] / (m || 1) * amp; i === 0 ? g.moveTo(x, y) : g.lineTo(x, y); }
      g.stroke();
    });
  }
  const motorTrace = () => motorShow.map(s => { const a = acts[epoch], n = neurons[s].node, out = new Float32Array(T); for (let i = 0; i < T; i++) out[i] = a[i * N + n]; return Array.from(out); });
  let motorCache = null, motorEpoch = -1;
  const glow = (obj, on) => { obj.material.opacity = on ? 1 : 0.4; };

  let last = performance.now();
  function frame(now) {
    requestAnimationFrame(frame);
    last = now;
    const e = meta.epochs[epoch];
    if (motorEpoch !== epoch) { motorCache = motorTrace(); motorEpoch = epoch; }
    // where are we in the staged run
    let name = 'idle', p = 0;
    if (stageT !== null) {
      let el = now - stageT; stageIdx = 0;
      while (el > STAGE[stageIdx].dur) { el -= STAGE[stageIdx].dur; stageIdx++; }
      name = STAGE[stageIdx].name; p = el / STAGE[stageIdx].dur;
    }
    const after = s => stageT !== null && stageIdx > STAGE.findIndex(x => x.name === s);
    const inT = name === 'input' ? Math.floor(p * (T - 1)) : after('input') ? T - 1 : -1;
    const brainT = name === 'brain' ? Math.floor(p * (T - 1)) : after('brain') ? T - 1 : 0;
    const gate = name === 'brain' ? 1 : after('brain') ? 0.25 : 0;
    const motorGlow = name === 'motor' ? Math.sin(Math.PI * Math.min(1, p)) : 0;
    activity(brainT, gate, motorGlow);
    hopProgress.fill(0);
    if (name === 'antenna') hopProgress[0] = p * 1.45;
    if (after('antenna')) hopProgress[0] = 2;
    if (name === 'brain') for (let h = 1; h <= maxHop; h++) hopProgress[h] = Math.max(0, (p * (maxHop + 0.6) - (h - 1))) ;
    if (after('brain')) for (let h = 1; h <= maxHop; h++) hopProgress[h] = 2;
    if (name === 'decoder') hopProgress[maxHop + 1] = p * 1.45;
    if (after('decoder')) hopProgress[maxHop + 1] = 2;
    if (name === 'output') hopProgress[maxHop + 2] = p * 1.45;
    if (after('output')) hopProgress[maxHop + 2] = 2;
    placeParticles(hopProgress);
    glow(inBlock, name === 'input' || name === 'antenna'); glow(inLink, name === 'antenna');
    glow(decBlock, name === 'decoder' || name === 'output'); glow(outLink, name === 'output'); glow(outBlock, name === 'output' || name === 'done');
    trace(plots.noisy, [e.noisy], ['#e8562a'], inT, [1.4]);
    trace(plots.motor, motorCache, ['#178f60'], name === 'motor' ? Math.floor(p * (T - 1)) : after('motor') ? T - 1 : -1, [1.1], 6);
    const outT = name === 'output' ? Math.floor(p * (T - 1)) : after('output') ? T - 1 : -1;
    trace(plots.out, [e.clean, e.decoded], ['#222222', '#178f60'], outT, [1.2, 1.8]);
    if (name === 'done' && !$('gain').textContent) $('gain').textContent = `+${e.snr_gain.toFixed(1)} dB cleaner`;
    $('stage').textContent = { idle: '', input: 'Input', antenna: 'Antenna', brain: 'Brain', motor: 'Motor', decoder: 'Decoder', output: 'Output', done: 'Done' }[name];
    key.position.copy(camera.position);
    controls.update();
    placeLabels();
    renderer.render(scene, camera);
  }
  requestAnimationFrame(frame);
  window.sim = { scene, camera, controls, regions, neurons, flows };
})();
