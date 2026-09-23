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
  const gltf = new THREE.GLTFLoader();
  const load = n => new Promise((res, rej) => gltf.load(`./models/${n}.glb${V}`, g => res(g.scene), null, rej));
  const [capGLB, pcbGLB, scopeGLB] = await Promise.all([load('cap'), load('pcb'), load('scope')]);
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
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new THREE.RoomEnvironment(), 0.04).texture;
  const key = new THREE.DirectionalLight(0xffffff, 1.5);
  const fill = new THREE.DirectionalLight(0xcfe0f5, 0.5);
  scene.add(key, fill, new THREE.AmbientLight(0xffffff, 0.18));

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
    uniforms: { act: { value: actTex }, base: { value: baseTex }, uFade: { value: 1 } }, transparent: true, depthWrite: false,
    vertexShader: `attribute float nid; uniform float uFade; uniform sampler2D act; uniform sampler2D base; varying vec4 vC;
      void main() {
        vec2 uv = vec2((nid + 0.5) / ${TEX_W}.0, 0.5);
        float a = texture2D(act, uv).r; vec4 b = texture2D(base, uv);
        vC = vec4(mix(vec3(0.60, 0.64, 0.72), b.rgb, smoothstep(0.0, 0.5, a)), b.a * (0.34 + 0.66 * a) * uFade);
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
  // Built in Blender (blender_hardware.py): a head in a 32 channel EEG cap, the decoder board, and
  // a bench oscilloscope. Sized against the brain so the fly stays the subject; real relative scale
  // would put a 0.5 mm brain next to a 200 mm head and you would see nothing.
  function place(obj, at, widthFrac, rot) {
    obj.updateMatrixWorld(true);
    const b = new THREE.Box3().setFromObject(obj), size = new THREE.Vector3();
    b.getSize(size);
    const k = (BW * widthFrac) / size.x;
    obj.scale.setScalar(k);
    obj.rotation.set(rot[0], rot[1], rot[2]);
    obj.updateMatrixWorld(true);
    const b2 = new THREE.Box3().setFromObject(obj), ctr = new THREE.Vector3();
    b2.getCenter(ctr);
    obj.position.add(at).sub(ctr);       // shift by (target - current centre); copying first doubled it
    obj.traverse(o => { if (o.isMesh) { o.castShadow = o.receiveShadow = false; } });
    scene.add(obj);
    return obj;
  }
  const CAP = new THREE.Vector3(BW * 0.82, -BH * 0.02, 0);
  const CHIP = new THREE.Vector3(-BW * 0.70, -BH * 0.24, 0);
  const SCOPE = new THREE.Vector3(-BW * 1.16, -BH * 0.06, 0);
  place(capGLB, CAP, 0.44, [0, -Math.PI * 0.72, 0]);
  place(pcbGLB, CHIP, 0.40, [-1.05, Math.PI, 0]);
  place(scopeGLB, SCOPE, 0.52, [0.08, Math.PI, 0]);

  // the two screens that actually show the signal: the monitor face, and the chip lid
  function screen(w, h) {
    const cv = document.createElement('canvas'); cv.width = 384; cv.height = 240;
    const tex = new THREE.CanvasTexture(cv);
    return { cv, g: cv.getContext('2d'), tex, w, h };
  }
  const monScreen = screen(512, 320);
  let screenMesh = null;
  scopeGLB.traverse(o => { if (o.isMesh && o.material && /screen/i.test(o.material.name)) screenMesh = o; });
  if (screenMesh) screenMesh.material = new THREE.MeshBasicMaterial({ map: monScreen.tex, toneMapped: false });
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
    const geo = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 140, BH * 0.011, 8);
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
  // the body of each model, found by the material Blender gave it, so a cable or a stand cannot
  // drag the anchor off the object
  function bodyBox(root, materialName) {
    root.updateMatrixWorld(true);
    const b = new THREE.Box3();
    root.traverse(o => { if (o.isMesh && o.material && o.material.name === materialName) b.expandByObject(o); });
    return b.isEmpty() ? new THREE.Box3().setFromObject(root) : b;
  }
  const capBox = bodyBox(capGLB, 'skin'), pcbBox = bodyBox(pcbGLB, 'solder_mask'), scopeBox = bodyBox(scopeGLB, 'case');
  const inWire = connector(new THREE.Vector3(capBox.min.x + BW * 0.04, capBox.min.y + BH * 0.18, 0), ANT, 0xe04f24, 0.16);
  const outWire = connector(new THREE.Vector3(pcbBox.min.x, CHIP.y, 0), new THREE.Vector3(scopeBox.max.x, SCOPE.y - BH * 0.10, 0), 0x0f9d63, 0.20);

  const motorSlots = neurons.map((n, s) => s).filter(s => neurons[s].group === 3 && group[neurons[s].node] !== 255);
  // The readout: every descending neuron reaches the board, but drawn as a gathered harness rather
  // than 1,400 straight lines converging on the camera's own target, which blinds the decoder shot.
  const GATHER = new THREE.Vector3(bbox.min.x - BW * 0.10, CHIP.y + BH * 0.16, 0);
  const fan = [], fanId = [];
  for (let k = 0; k < motorSlots.length; k += 9) {
    const sl = motorSlots[k], n = neurons[sl].node;
    const p = toScene(new Float32Array([pos[3 * n], pos[3 * n + 1], pos[3 * n + 2]]));
    const a = new THREE.Vector3(p[0], p[1], p[2]);
    const g = GATHER.clone().add(new THREE.Vector3(0, (Math.random() - 0.5) * BH * 0.10, (Math.random() - 0.5) * BH * 0.10));
    const b = new THREE.Vector3(pcbBox.max.x, CHIP.y + (Math.random() - 0.5) * BH * 0.12, (Math.random() - 0.5) * BH * 0.08);
    const curve = new THREE.QuadraticBezierCurve3(a, g, b).getPoints(10);
    for (let i = 0; i < curve.length - 1; i++) {
      fan.push(curve[i].x, curve[i].y, curve[i].z, curve[i + 1].x, curve[i + 1].y, curve[i + 1].z);
      fanId.push(sl, sl);
    }
  }
  const fanGeom = new THREE.BufferGeometry();
  fanGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(fan), 3));
  fanGeom.setAttribute('nid', new THREE.BufferAttribute(new Float32Array(fanId), 1));
  const fanMat = lineMat.clone();                    // its own fade, so the readout survives the arbors dimming
  fanMat.uniforms.act.value = actTex; fanMat.uniforms.base.value = baseTex;
  scene.add(new THREE.LineSegments(fanGeom, fanMat));

  // ---- labels: one or two words, nothing else ----
  const labelBox = $('labels'), labels = [];
  function label(text, at, cls = 'station', colour = '') {
    const d = document.createElement('div'); d.textContent = text; d.className = cls; if (colour) d.style.color = colour;
    labelBox.appendChild(d); const L = { d, at, alpha: 1 }; labels.push(L); return L;
  }
  const cc = new THREE.Vector3(), pc = new THREE.Vector3(), sc2 = new THREE.Vector3();
  capBox.getCenter(cc); pcbBox.getCenter(pc); scopeBox.getCenter(sc2);
  label('EEG cap', new THREE.Vector3(cc.x, capBox.max.y + BH * 0.12, cc.z), 'station', '#e04f24');
  const antLab = label('Antenna', ANT.clone().add(new THREE.Vector3(0, -BH * 0.16, 0)), 'station', '#e0921b');
  label('Decoder', new THREE.Vector3(pc.x, pcbBox.max.y + BH * 0.12, pc.z), 'station', '#10141c');
  label('Output', new THREE.Vector3(sc2.x, scopeBox.max.y + BH * 0.12, sc2.z), 'station', '#0f9d63');
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
  const centre = new THREE.Vector3((CAP.x + SCOPE.x) / 2, -BH * 0.04, 0);
  const SHOT = {
    wide: { at: centre, d: 2.25, el: 0.15, az: 0 },
    input: { at: CAP.clone().lerp(ANT, 0.28), d: 0.95, el: 0.13, az: -0.16 },
    antenna: { at: ANT.clone().lerp(new THREE.Vector3(0, 0, 0), 0.45), d: 1.35, el: 0.12, az: -0.22 },
    brain: { at: new THREE.Vector3(0, -BH * 0.02, 0), d: 1.45, el: 0.21, az: 0.15 },
    motor: { at: new THREE.Vector3(CHIP.x * 0.55, CHIP.y + BH * 0.24, 0), d: 1.15, el: 0.26, az: -0.12 },
    decoder: { at: CHIP.clone(), d: 0.72, el: 0.26, az: -0.20 },
    output: { at: SCOPE.clone(), d: 0.62, el: 0.12, az: -0.08 },
  };
  const SPAN = CAP.distanceTo(SCOPE);
  // the wide shot is measured, not guessed: fit the whole chain, both axes, with a margin
  function fitDistance(margin = 1.03) {
    const b = new THREE.Box3();
    [capGLB, pcbGLB, scopeGLB].forEach(o => b.expandByObject(o));
    b.expandByPoint(new THREE.Vector3(bbox.min.x, bbox.min.y, bbox.min.z));
    b.expandByPoint(new THREE.Vector3(bbox.max.x, bbox.max.y, bbox.max.z));
    const size = new THREE.Vector3(); b.getSize(size);
    b.getCenter(SHOT.wide.at);
    const vFov = THREE.MathUtils.degToRad(camera.fov), aspect = Math.max(view.clientWidth / view.clientHeight, 0.6);
    const dv = (size.y / 2) / Math.tan(vFov / 2);
    const dh = (size.x / 2) / (Math.tan(vFov / 2) * aspect);
    return (Math.max(dv, dh) + size.z / 2) * margin;
  }
  const camAt = new THREE.Vector3();
  const shotPos = (s, o) => {
    const d = s === SHOT.wide ? fitDistance() : s.d * SPAN * 0.56;
    return o.set(s.at.x + Math.sin(s.az) * d, s.at.y + Math.sin(s.el) * d, s.at.z - Math.cos(s.az) * Math.cos(s.el) * d);
  };
  let userCam = false, shotFrom = null, shotTo = SHOT.wide, shotStart = -1e9;
  function setShot(s) { if (!s || s === shotTo) return; shotFrom = { pos: camera.position.clone(), tgt: controls.target.clone() }; shotTo = s; shotStart = performance.now(); }
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
      if (k === 'antenna') v = ant ? Math.min(1, p * 1.4) : 0.26;
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
    const lanes = [[], []];                                  // two label rows, so close ticks never collide
    BENCH.forEach((b, i) => {
      const x = X(b.v), up = i % 2 === 0 ? 0 : 1;
      g.strokeStyle = b.tone; g.lineWidth = 2;
      g.beginPath(); g.moveTo(x, y - (up ? 9 : 0)); g.lineTo(x, y + (up ? 0 : 9)); g.stroke();
      const w = g.measureText(b.name).width;
      let tx = Math.min(Math.max(x, L + w / 2), R - w / 2);
      if (lanes[up].some(o => Math.abs(o - tx) < w + 6)) tx = Math.min(R - w / 2, tx + w / 2 + 8);
      lanes[up].push(tx);
      g.fillStyle = b.tone; g.textAlign = 'center';
      g.fillText(b.name, tx, up ? y - 16 : y + 17);
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
  function frame() {
    requestAnimationFrame(frame);
    const now = performance.now();
    const dt = Math.min(now - last, 100); last = now;
    const e = meta.epochs[epoch];
    if (motorEpoch !== epoch) { motorCache = motorTrace(); motorEpoch = epoch; }
    if (playing) { elapsed = now - origin; if (elapsed >= TOTAL) { elapsed = TOTAL; playing = false; syncButtons(); } }

    let k = 'idle', p = 0, i = -1;
    if (elapsed >= 0) {
      let el = elapsed; i = 0;
      while (i < STAGE.length && el >= STAGE[i].dur) { el -= STAGE[i].dur; i++; }
      if (i >= STAGE.length) { k = 'done'; i = STAGE.length - 1; p = 1; }
      else { k = STAGE[i].key; p = el / STAGE[i].dur; }
    }
    const started = kk => elapsed >= 0 && STAGE.findIndex(s => s.key === kk) <= i;
    const upto = kk => k === kk ? Math.floor(p * (T - 1)) : started(kk) ? T - 1 : -1;

    const state = k === 'done' ? 'done' : i;
    if (state !== idx) {
      idx = state;
      STAGE.forEach((s, j) => { s.el.classList.toggle('on', j === i && k !== 'done'); s.el.classList.toggle('done', j < i || k === 'done'); });
      rail.classList.toggle('finished', k === 'done');
      if (k === 'done') STAGE.forEach(st => st.fill.style.width = '100%');
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
    const late = k === 'decoder' || k === 'output' || k === 'done';
    hullMat.uniforms.uDim.value = late ? 1 : 0;
    lineMat.uniforms.uFade.value += ((late ? 0.10 : 1) - lineMat.uniforms.uFade.value) * Math.min(1, dt / 260);
    fanMat.uniforms.uFade.value += ((late ? 0.55 : 1) - fanMat.uniforms.uFade.value) * Math.min(1, dt / 260);
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
      const shotT = reduce ? 1 : Math.min(1, (now - shotStart) / 1100);
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
  window.sim = { scene, camera, controls, SHOT, neurons, segments: keep.length,
    debug: () => ({ shot: Object.keys(SHOT).find(k => SHOT[k] === shotTo), userCam, shotStart, now: performance.now() }) };
})();
