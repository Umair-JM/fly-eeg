// Fly brain EEG cleaner: plays back recorded activity of the MaleCNS central brain (every neuron,
// every EEG sample) for real test epochs, exported by export_sim.py into sim/data/.
(async function () {
  const DATA = './data/';
  const status = document.getElementById('status');
  const [meta, pos, group] = await Promise.all([
    fetch(DATA + 'meta.json').then(r => r.json()),
    fetch(DATA + 'pos.bin').then(r => r.arrayBuffer()).then(b => new Float32Array(b)),
    fetch(DATA + 'group.bin').then(r => r.arrayBuffer()).then(b => new Uint8Array(b)),
  ]);
  const N = meta.n, T = meta.t;
  const acts = await Promise.all(meta.epochs.map((_, k) => fetch(DATA + `act_${k}.bin`).then(r => r.arrayBuffer()).then(b => new Uint8Array(b))));
  status.textContent = `${N.toLocaleString()} neurons, ${T} samples per epoch at ${meta.fs} Hz, ${meta.epochs.length} epochs loaded`;

  // ---- 3D scene: one point per neuron, size 0 hides neurons without a soma position ----
  const view = document.getElementById('view');
  const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: false });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  view.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0e14);
  const camera = new THREE.PerspectiveCamera(45, 1, 1, 5000);

  const c = [0, 0, 0];
  let n_pos = 0;
  for (let i = 0; i < N; i++) if (group[i] !== 255) { c[0] += pos[3 * i]; c[1] += pos[3 * i + 1]; c[2] += pos[3 * i + 2]; n_pos++; }
  c[0] /= n_pos; c[1] /= n_pos; c[2] /= n_pos;
  const xyz = new Float32Array(3 * N), size = new Float32Array(N), color = new Float32Array(3 * N);
  const BASE = { 0: [0.20, 0.30, 0.48], 1: [1.0, 0.70, 0.28], 2: [0.78, 0.57, 0.92], 3: [0.24, 0.86, 0.59] };
  for (let i = 0; i < N; i++) {
    xyz[3 * i] = pos[3 * i] - c[0];
    xyz[3 * i + 1] = -(pos[3 * i + 1] - c[1]);      // EM y grows ventrally; screen up = dorsal
    xyz[3 * i + 2] = pos[3 * i + 2] - c[2];
    size[i] = group[i] === 255 ? 0 : group[i] === 0 ? 3.0 : 4.2;
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(xyz, 3));
  geom.setAttribute('size', new THREE.BufferAttribute(size, 1));
  geom.setAttribute('color', new THREE.BufferAttribute(color, 3));
  const mat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: `attribute float size; attribute vec3 color; varying vec3 vC;
      void main() { vC = color; vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = size * (320.0 / -mv.z) * ${Math.min(devicePixelRatio, 2).toFixed(1)}; gl_Position = projectionMatrix * mv; }`,
    fragmentShader: `varying vec3 vC; void main() { float d = length(gl_PointCoord - 0.5);
        if (d > 0.5) discard; float a = smoothstep(0.5, 0.1, d); gl_FragColor = vec4(vC * a, a); }`,
  });
  scene.add(new THREE.Points(geom, mat));
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.autoRotate = true; controls.autoRotateSpeed = 0.6; controls.enableDamping = true;
  camera.position.set(0, 100, 640);
  function resize() {
    const w = view.clientWidth, h = view.clientHeight;
    renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  addEventListener('resize', resize); resize();

  // ---- panel ----
  const $ = id => document.getElementById(id);
  const plots = { noisy: $('noisy'), raster: $('raster'), out: $('out') };
  for (const k in plots) { plots[k].width = 800; plots[k].height = k === 'raster' ? 260 : 192; }
  const readoutIdx = []; for (let i = 0; i < N; i++) if (group[i] === 3) readoutIdx.push(i);
  const rasterRows = 130, rasterPick = Array.from({ length: rasterRows }, (_, r) => readoutIdx[Math.floor(r * readoutIdx.length / rasterRows)]);
  const rasterImg = plots.raster.getContext('2d').createImageData(T, rasterRows);

  let epoch = 0, t = 0, playing = true, speed = 2, pos_s = 0;     // pos_s: playhead in samples, advanced by wall clock
  const epochBox = $('epochs');
  const labels = ['hard', 'typical', 'mild'];
  meta.epochs.forEach((e, k) => {
    const b = document.createElement('button');
    b.textContent = `${labels[k] || 'epoch ' + k}: ${e.snr_in.toFixed(1)} dB in`;
    b.onclick = () => setEpoch(k);
    epochBox.appendChild(b);
  });
  function setEpoch(k) {
    epoch = k; t = 0; pos_s = 0;
    [...epochBox.children].forEach((b, i) => b.classList.toggle('on', i === k));
    const e = meta.epochs[k];
    $('snrin').textContent = e.snr_in.toFixed(1);
    $('m_in').textContent = e.snr_in.toFixed(1) + ' dB';
    $('m_gain').textContent = '+' + e.snr_gain.toFixed(1) + ' dB';
    $('m_cc').textContent = e.cc.toFixed(3);
    $('m_rrmse').textContent = e.rrmse.toFixed(3);
    buildRaster();
  }
  function buildRaster() {
    const a = acts[epoch], d = rasterImg.data;
    for (let r = 0; r < rasterRows; r++) for (let s = 0; s < T; s++) {
      const v = a[s * N + rasterPick[r]] / 255, o = 4 * (r * T + s);
      d[o] = 20 + 40 * v; d[o + 1] = 30 + 200 * v; d[o + 2] = 40 + 120 * v; d[o + 3] = 255;
    }
  }
  $('play').onclick = () => { playing = !playing; $('play').textContent = playing ? 'Pause' : 'Play'; };
  $('speed').oninput = ev => speed = +ev.target.value;
  setEpoch(1);

  function trace(cv, series, colors, upto, lineW) {
    const g = cv.getContext('2d'), W = cv.width, H = cv.height;
    g.clearRect(0, 0, W, H);
    let m = 0; for (const s of series) for (let i = 0; i < T; i++) m = Math.max(m, Math.abs(s[i]));
    g.strokeStyle = '#232a3a'; g.beginPath(); g.moveTo(0, H / 2); g.lineTo(W, H / 2); g.stroke();
    series.forEach((s, j) => {
      g.strokeStyle = colors[j]; g.lineWidth = lineW[j]; g.beginPath();
      for (let i = 0; i <= upto; i++) { const x = i * W / (T - 1), y = H / 2 - s[i] / m * (H / 2 - 6); i ? g.lineTo(x, y) : g.moveTo(x, y); }
      g.stroke();
      g.globalAlpha = 0.22; g.beginPath();
      for (let i = upto; i < T; i++) { const x = i * W / (T - 1), y = H / 2 - s[i] / m * (H / 2 - 6); i === upto ? g.moveTo(x, y) : g.lineTo(x, y); }
      g.stroke(); g.globalAlpha = 1;
    });
    g.strokeStyle = '#8a93a6'; g.lineWidth = 1; g.beginPath(); g.moveTo(upto * W / (T - 1), 0); g.lineTo(upto * W / (T - 1), H); g.stroke();
  }
  function drawRaster(upto) {
    const cv = plots.raster, g = cv.getContext('2d');
    g.putImageData(rasterImg, 0, 0, 0, 0, upto + 1, rasterRows);
    g.fillStyle = '#0e121b'; g.fillRect(upto + 1, 0, T, rasterRows);
    g.strokeStyle = '#8a93a6'; g.beginPath(); g.moveTo(upto, 0); g.lineTo(upto, rasterRows); g.stroke();
  }

  function paintNeurons(sample) {
    const a = acts[epoch], off = sample * N;
    for (let i = 0; i < N; i++) {
      const b = BASE[group[i]] || BASE[0], v = a[off + i] / 255;
      const k = 0.45 + 1.6 * v, w = 0.9 * v * v;      // rest = dim base colour, active = bright and whiter
      color[3 * i] = b[0] * k + w; color[3 * i + 1] = b[1] * k + w; color[3 * i + 2] = b[2] * k + w;
    }
    geom.attributes.color.needsUpdate = true;
  }

  let last = performance.now();
  function frame(now) {
    requestAnimationFrame(frame);
    const dt = now - last; last = now;
    if (playing) { pos_s = (pos_s + speed * 60 * dt / 1000) % T; t = Math.floor(pos_s); }   // speed 1 = 60 samples/s
    const e = meta.epochs[epoch];
    paintNeurons(t);
    trace(plots.noisy, [e.noisy], ['#ff7a59'], t, [1.4]);
    drawRaster(t);
    trace(plots.out, [e.clean, e.decoded], ['#ffffff', '#3ddc97'], Math.max(0, t - meta.lag), [1.2, 1.6]);
    $('clock').textContent = `sample ${t} / ${T}  (${(t / meta.fs).toFixed(2)} s)`;
    controls.update();
    renderer.render(scene, camera);
  }
  requestAnimationFrame(frame);
})();
