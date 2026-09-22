// Fly brain EEG cleaner. Plays back the recorded activity of the MaleCNS central brain (every neuron,
// every EEG sample; export_sim.py) on the official neuron skeletons and neuropil meshes
// (export_anatomy.py). Antenna in, brain regions glow as the signal passes, motor neurons out, decoder.
(async function () {
  const DATA = './data/';
  const bin = (name, T) => fetch(DATA + name).then(r => r.arrayBuffer()).then(b => new T(b));
  const [meta, anat, skel, skelId, shellBuf, pos, group] = await Promise.all([
    fetch(DATA + 'meta.json').then(r => r.json()), fetch(DATA + 'anatomy.json').then(r => r.json()),
    bin('skel.bin', Float32Array), bin('skel_id.bin', Uint32Array), fetch(DATA + 'shell.bin').then(r => r.arrayBuffer()),
    bin('pos.bin', Float32Array), bin('group.bin', Uint8Array),
  ]);
  const acts = await Promise.all(meta.epochs.map((_, k) => bin(`act_${k}.bin`, Uint8Array)));
  const OPTIC = new Set(['ME', 'LO', 'LOP', 'LA', 'AME', 'CV-anterior', 'CRN']);
  const N = meta.n, T = meta.t, neurons = anat.neurons, S = neurons.length;
  const regions = anat.regions.map((r, k) => ({ ...r, k, optic: OPTIC.has(r.label) }));
  document.getElementById('loading').remove();

  // ---- coordinates: micrometres, centred on the brain shell, EM y flipped so dorsal is up ----
  const hdr = new Uint32Array(shellBuf, 0, 2), NV = hdr[0], NF = hdr[1];
  const shellV = new Float32Array(shellBuf, 8, NV * 3), shellF = new Uint32Array(shellBuf, 8 + NV * 12, NF * 3);
  const c = [0, 0, 0]; let nc = 0, yMax = -1e9, zMax = -1e9;
  for (const r of regions) { if (r.optic) continue; for (let i = r.v0; i < r.v0 + r.nv; i++) { c[0] += shellV[3 * i]; c[1] += shellV[3 * i + 1]; c[2] += shellV[3 * i + 2]; nc++; yMax = Math.max(yMax, shellV[3 * i + 1]); zMax = Math.max(zMax, shellV[3 * i + 2]); } }
  c[0] /= nc; c[1] /= nc; c[2] /= nc;
  const toScene = (arr) => { const o = new Float32Array(arr.length); for (let i = 0; i < arr.length; i += 3) { o[i] = arr[i] - c[0]; o[i + 1] = -(arr[i + 1] - c[1]); o[i + 2] = arr[i + 2] - c[2]; } return o; };

  const view = document.getElementById('view');
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  view.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x07090f);
  const camera = new THREE.PerspectiveCamera(40, 1, 1, 6000);

  // ---- neuropil shell: one translucent mesh per region so a region can glow on its own ----
  const shellPos = new THREE.BufferAttribute(toScene(shellV), 3);
  const regionMesh = regions.map(r => {
    if (r.optic) return null;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', shellPos);
    g.setIndex(new THREE.BufferAttribute(shellF.subarray(r.f0 * 3, (r.f0 + r.nf) * 3), 1));
    const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: 0x3b4d6e, transparent: true, opacity: 0.07, depthWrite: false, side: THREE.DoubleSide }));
    scene.add(m);
    return m;
  });

  // ---- text sprites (region names, antenna, decoder, output): one or two words each ----
  function label(text, size = 22, color = '#dfe6f2', scale = 120) {
    const cv = document.createElement('canvas'); cv.width = 256; cv.height = 64;
    const g = cv.getContext('2d'); g.font = `600 ${size}px system-ui, Segoe UI, sans-serif`; g.textAlign = 'center'; g.textBaseline = 'middle';
    g.fillStyle = color; g.fillText(text, 128, 32);
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(cv), transparent: true, depthWrite: false, depthTest: false }));
    sp.scale.set(scale, scale / 4, 1); sp.renderOrder = 10;      // drawn after the neurons so text is never buried
    scene.add(sp);
    return sp;
  }
  const regionLabel = regions.map(r => { if (r.optic) return null; const s = label(r.label, 24); const p = toScene(new Float32Array(r.centre)); s.position.set(p[0], p[1], p[2]); s.material.opacity = 0; return s; });

  // ---- neurons: official skeletons as line segments; colour per neuron comes from two 1-D textures ----
  const TEX_W = 4096;
  const actData = new Uint8Array(TEX_W * 4), baseData = new Uint8Array(TEX_W * 4);
  const BASE = { 0: [0.50, 0.70, 1.0], 1: [1.0, 0.70, 0.28], 2: [1.0, 0.70, 0.28], 3: [0.24, 0.90, 0.60] };
  neurons.forEach((n, s) => { const b = BASE[n.group]; baseData[4 * s] = b[0] * 255; baseData[4 * s + 1] = b[1] * 255; baseData[4 * s + 2] = b[2] * 255; baseData[4 * s + 3] = n.group === 1 || n.group === 2 ? 255 : 110; });
  const actTex = new THREE.DataTexture(actData, TEX_W, 1, THREE.RGBAFormat), baseTex = new THREE.DataTexture(baseData, TEX_W, 1, THREE.RGBAFormat);
  baseTex.needsUpdate = true;
  const lineMat = new THREE.ShaderMaterial({
    uniforms: { act: { value: actTex }, base: { value: baseTex } }, transparent: true, depthWrite: false,
    vertexShader: `attribute float nid; uniform sampler2D act; uniform sampler2D base; varying vec4 vC;
      void main() { vec2 uv = vec2((nid + 0.5) / ${TEX_W}.0, 0.5); float a = texture2D(act, uv).r; vec4 b = texture2D(base, uv);
        vC = vec4(b.rgb * (0.35 + 1.1 * a) + vec3(0.8 * a * a), b.a * (0.25 + 0.75 * a)); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: `varying vec4 vC; void main() { gl_FragColor = vC; }`,
  });
  const keepSeg = []; const yCut = yMax + 30, zCut = zMax + 30;   // beyond the brain shell (ventral y, posterior z) = neck and body
  const drawn = neurons.map((n, s) => n.group !== 3 || s % 4 === 0);
  for (let i = 0; i < skel.length; i += 6) if (drawn[skelId[i / 3]] && skel[i + 1] < yCut && skel[i + 4] < yCut && skel[i + 2] < zCut && skel[i + 5] < zCut) keepSeg.push(i / 6);
  const skelPos = new Float32Array(keepSeg.length * 6), skelNid = new Float32Array(keepSeg.length * 2);
  keepSeg.forEach((sg, j) => { for (let q = 0; q < 6; q++) skelPos[6 * j + q] = skel[6 * sg + q]; skelNid[2 * j] = skelId[2 * sg]; skelNid[2 * j + 1] = skelId[2 * sg + 1]; });
  const skelGeom = new THREE.BufferGeometry();
  skelGeom.setAttribute('position', new THREE.BufferAttribute(toScene(skelPos), 3));
  skelGeom.setAttribute('nid', new THREE.BufferAttribute(skelNid, 1));
  scene.add(new THREE.LineSegments(skelGeom, lineMat));

  // antenna labels: at the outermost point of each antenna's nerve (where the signal enters)
  const P = skelGeom.attributes.position.array, far = { 1: [0, 0, 0, -1], 2: [0, 0, 0, -1] };
  for (let i = 0; i < skelNid.length; i++) {
    const g = neurons[skelNid[i]].group;
    if (g !== 1 && g !== 2) continue;
    const d = P[3 * i] ** 2 + P[3 * i + 1] ** 2 + P[3 * i + 2] ** 2;
    if (d > far[g][3]) far[g] = [P[3 * i], P[3 * i + 1], P[3 * i + 2], d];
  }
  for (const g of [1, 2]) { const a = far[g]; label('Antenna', 26, '#ffb347', 170).position.set(a[0] * 1.08, a[1] * 1.08 - 25, a[2] * 1.08); }

  // ---- decoder: a box to the right, wired to a sample of motor neurons; output beyond it ----
  const DEC = new THREE.Vector3(-360, -40, 0);
  const box = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(50, 50, 50)), new THREE.LineBasicMaterial({ color: 0x3ddc97 }));
  box.position.copy(DEC); scene.add(box);
  label('Decoder', 26, '#3ddc97', 180).position.set(DEC.x, DEC.y + 50, DEC.z);
  const outLabel = label('Output', 26, '#3ddc97', 180); outLabel.position.set(DEC.x - 165, DEC.y, DEC.z);
  scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(DEC.x - 25, DEC.y, DEC.z), new THREE.Vector3(DEC.x - 110, DEC.y, DEC.z)]), new THREE.LineBasicMaterial({ color: 0x3ddc97 })));
  const motorSlots = neurons.map((n, s) => s).filter(s => neurons[s].group === 3 && group[neurons[s].node] !== 255);
  const wires = [], wireId = [];
  for (let k = 0; k < motorSlots.length; k += 4) {
    if (!drawn[motorSlots[k]]) continue;
    const s = motorSlots[k], n = neurons[s].node, p = toScene(new Float32Array([pos[3 * n], pos[3 * n + 1], pos[3 * n + 2]]));
    wires.push(p[0], p[1], p[2], DEC.x + 25, DEC.y + (Math.random() - 0.5) * 40, DEC.z + (Math.random() - 0.5) * 40); wireId.push(s, s);
  }
  const wireGeom = new THREE.BufferGeometry();
  wireGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(wires), 3));
  wireGeom.setAttribute('nid', new THREE.BufferAttribute(new Float32Array(wireId), 1));
  scene.add(new THREE.LineSegments(wireGeom, lineMat));

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.autoRotate = true; controls.autoRotateSpeed = 0.5; controls.enableDamping = true;
  controls.target.set(-110, -20, 0);
  camera.position.set(-110, 80, -1250);
  function resize() { const w = view.clientWidth, h = view.clientHeight; renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix(); }
  addEventListener('resize', resize); resize();
  let down = null;                                      // a click (no drag) stops or resumes the rotation
  view.addEventListener('pointerdown', e => down = [e.clientX, e.clientY]);
  view.addEventListener('pointerup', e => { if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) < 5) controls.autoRotate = !controls.autoRotate; down = null; });

  // ---- per-neuron and per-region activity ----
  const stats = acts.map(a => {
    const mu = new Float32Array(S), sd = new Float32Array(S);
    for (let t = 0; t < T; t++) { const o = t * N; for (let s = 0; s < S; s++) mu[s] += a[o + neurons[s].node]; }
    for (let s = 0; s < S; s++) mu[s] /= T;
    for (let t = 0; t < T; t++) { const o = t * N; for (let s = 0; s < S; s++) { const d = a[o + neurons[s].node] - mu[s]; sd[s] += d * d; } }
    for (let s = 0; s < S; s++) sd[s] = Math.sqrt(sd[s] / T);
    return { mu, sd };
  });
  const regW = regions.map(() => []);                   // per region: [slot, weight] from the skeleton occupancy
  neurons.forEach((n, s) => { if (n.group === 1 || n.group === 2) return; for (const r in n.regions) regW[r].push([s, n.regions[r]]); });
  const dev = new Float32Array(S), regAct = new Float32Array(regions.length);
  function activity(sample) {
    const a = acts[epoch], { mu, sd } = stats[epoch], off = sample * N;
    for (let s = 0; s < S; s++) {
      const n = neurons[s], raw = a[off + n.node];
      dev[s] = n.group === 1 || n.group === 2 ? raw / 255 : Math.min(1, Math.max(0, (raw - mu[s]) / (2.5 * sd[s] + 3)));
      actData[4 * s] = dev[s] * 255;
    }
    actTex.needsUpdate = true;
    regions.forEach((r, k) => {
      if (r.optic) return;
      let num = 0, den = 0;
      for (const [s, w] of regW[k]) { num += w * dev[s]; den += w; }
      const v = den ? Math.min(1, 2.2 * num / den) : 0;
      regAct[k] = v;
      const m = regionMesh[k].material;
      m.opacity = 0.06 + 0.45 * v; m.color.setRGB(0.23 + 0.7 * v, 0.30 + 0.55 * v, 0.43 + 0.3 * v);
      regionLabel[k].material.opacity = v > 0.25 ? Math.min(1, (v - 0.25) * 3) : 0;
    });
  }

  // ---- panel ----
  const $ = id => document.getElementById(id);
  const plots = { noisy: $('noisy'), motor: $('motor'), out: $('out') };
  for (const k in plots) { plots[k].width = 800; plots[k].height = k === 'motor' ? 300 : 220; }
  const motorShow = motorSlots.slice().sort((a, b) => stats[0].sd[b] - stats[0].sd[a]).slice(0, 6);   // six lively motor neurons
  let epoch = 0, t = 0, playing = true, speed = 2, pos_s = 0;
  const epochBox = $('epochs'), labels = ['hard', 'typical', 'mild'];
  meta.epochs.forEach((e, k) => { const b = document.createElement('button'); b.textContent = labels[k] || 'epoch ' + k; b.onclick = () => setEpoch(k); epochBox.appendChild(b); });
  function setEpoch(k) {
    epoch = k; t = 0; pos_s = 0;
    [...epochBox.children].forEach((b, i) => b.classList.toggle('on', i === k));
    const e = meta.epochs[k];
    $('gain').textContent = `+${e.snr_gain.toFixed(1)} dB cleaner`;
  }
  $('play').onclick = () => { playing = !playing; $('play').textContent = playing ? 'Pause' : 'Play'; };
  $('speed').oninput = ev => speed = +ev.target.value;
  setEpoch(1);

  function trace(cv, series, colors, upto, lineW, rows = 1) {
    const g = cv.getContext('2d'), W = cv.width, H = cv.height, h = H / rows;
    g.clearRect(0, 0, W, H);
    series.forEach((s, j) => {
      let m = 0; for (let i = 0; i < T; i++) m = Math.max(m, Math.abs(s[i]));
      if (rows > 1) { let mu = 0; for (let i = 0; i < T; i++) mu += s[i]; mu /= T; m = 0; for (let i = 0; i < T; i++) m = Math.max(m, Math.abs(s[i] - mu)); s = s.map(v => v - mu); }
      const y0 = rows > 1 ? h * (j + 0.5) : H / 2, amp = rows > 1 ? h / 2 - 3 : H / 2 - 6;
      for (const [a, b, alpha] of [[0, upto, 1], [upto, T - 1, 0.22]]) {
        g.globalAlpha = alpha; g.strokeStyle = colors[j % colors.length]; g.lineWidth = lineW[j % lineW.length]; g.beginPath();
        for (let i = a; i <= b; i++) { const x = i * W / (T - 1), y = y0 - s[i] / (m || 1) * amp; i === a ? g.moveTo(x, y) : g.lineTo(x, y); }
        g.stroke();
      }
    });
    g.globalAlpha = 1; g.strokeStyle = '#8a93a6'; g.lineWidth = 1; g.beginPath(); g.moveTo(upto * W / (T - 1), 0); g.lineTo(upto * W / (T - 1), H); g.stroke();
  }
  const motorTrace = () => motorShow.map(s => { const a = acts[epoch], n = neurons[s].node, out = new Float32Array(T); for (let i = 0; i < T; i++) out[i] = a[i * N + n]; return Array.from(out); });
  let motorCache = null, motorEpoch = -1;

  let last = performance.now();
  function frame(now) {
    requestAnimationFrame(frame);
    const dt = now - last; last = now;
    if (playing) { pos_s = (pos_s + speed * 60 * dt / 1000) % T; t = Math.floor(pos_s); }
    const e = meta.epochs[epoch];
    activity(t);
    trace(plots.noisy, [e.noisy], ['#ff7a59'], t, [1.4]);
    if (motorEpoch !== epoch) { motorCache = motorTrace(); motorEpoch = epoch; }
    trace(plots.motor, motorCache, ['#3ddc97'], t, [1.1], 6);
    trace(plots.out, [e.clean, e.decoded], ['#ffffff', '#3ddc97'], Math.max(0, t - meta.lag), [1.2, 1.6]);
    $('clock').textContent = `${(t / meta.fs).toFixed(2)} s`;
    controls.update();
    renderer.render(scene, camera);
  }
  requestAnimationFrame(frame);
  window.sim = { scene, camera, controls, far, regions, neurons };
})();
