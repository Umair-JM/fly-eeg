"""Fly-brain EEG denoiser: the MaleCNS connectome as a fixed reservoir. Noisy EEG goes in through
the Johnston's organ (antennal vibration) neurons, clean EEG is read out from descending + motor
neurons by a ridge regression. Nothing inside the brain is trained.

Benchmark: EEGdenoiseNet (Zhang et al. 2021), semi-synthetic EOG / EMG contamination, the same
protocol SPAR-EEG (Shaikh et al. 2026, TNSRE) is evaluated on.

Run from the nfly folder:  uv run --no-sync python "../fly-eeg/fly_eeg_denoise.py" --artifact eog
"""
import argparse, time, urllib.request
from pathlib import Path
import numpy as np, torch
from nfly.connectome import load_malecns, build_connectome
from nfly.brain import ConnectomeRNN, InputDrive

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
NFLY_DATA = HERE.parent / "nfly" / "data"
RAW = "https://raw.githubusercontent.com/ncclabsustech/EEGdenoiseNet/master/data/"
FS = 256  # EEGdenoiseNet sampling rate
CENTRAL = ["cb_intrinsic", "descending_neuron", "cb_motor"]


def fetch(name):
    DATA.mkdir(exist_ok=True)
    p = DATA / name
    if not p.exists():
        print("downloading", name, flush=True)
        urllib.request.urlretrieve(RAW + name, p)
    return np.load(p)


def contaminate(eeg, art, snr_db, rng):
    """EEGdenoiseNet protocol: y = x + lam * n with lam set by the target SNR; both scaled by std(y)."""
    i = rng.integers(len(art), size=len(eeg))
    n = art[i]
    lam = eeg.std(1, keepdims=True) / (n.std(1, keepdims=True) * 10 ** (snr_db[:, None] / 20))
    y = eeg + lam * n
    s = y.std(1, keepdims=True)
    return (eeg / s).astype(np.float32), (y / s).astype(np.float32)


LEVELS = np.arange(-20, 6)          # SPAR-EEG nominal input SNR grid (dB)


def load_centered(a, rng, eeg, eog, emg):
    """SPAR-EEG protocol (Shaikh et al. 2026, prepare_datasets_centered.py): records paired by
    index, the artifact occupies only the middle third of the epoch, the clean EEG has unit std
    inside that window and the SNR is defined inside it. Train / val at SNR ~ U(-20, 5) dB;
    the same test records at every integer level. Returns x, y and the per-epoch nominal SNR."""
    T = eeg.shape[1]
    lo, hi = T // 3, 2 * (T // 3)
    n = min(len(eeg), len(eog), len(emg))
    idx = rng.permutation(n)

    def mix(i, snr):
        clean = eeg[i] / eeg[i][:, lo:hi].std(1, keepdims=True)
        parts = {"eog": eog[i][:, lo:hi], "emg": emg[i][:, lo:hi]}
        if a.artifact == "both":
            art = parts["eog"] / parts["eog"].std(1, keepdims=True) + parts["emg"] / parts["emg"].std(1, keepdims=True)
        else:
            art = parts[a.artifact]
        noise = np.zeros_like(clean)
        noise[:, lo:hi] = art / art.std(1, keepdims=True) * np.sqrt(10 ** (-snr[:, None] / 10))
        y = clean + noise
        sc = y.std(1, keepdims=True)
        return (clean / sc).astype(np.float32), (y / sc).astype(np.float32)

    n_va = a.n_train // 5
    x, y, snr = {}, {}, {}
    for name, sl in [("tr", idx[:a.n_train - n_va]), ("va", idx[a.n_train - n_va:a.n_train])]:
        snr[name] = rng.uniform(-20, 5, size=len(sl))
        x[name], y[name] = mix(sl, snr[name])
    te = idx[a.n_train:a.n_train + a.n_test]
    snr["te"] = np.repeat(LEVELS, len(te)).astype(float)
    x["te"], y["te"] = (np.concatenate(v) for v in zip(*[mix(te, np.full(len(te), float(l))) for l in LEVELS]))
    return x, y, snr


def load_data(a, rng):
    """x (clean) and y (noisy) dicts with tr / va / te splits, SNR uniform in [-7, 2] dB."""
    eeg = fetch("EEG_all_epochs.npy")
    art = {"eog": fetch("EOG_all_epochs.npy"), "emg": fetch("EMG_all_epochs.npy")}
    if a.protocol == "centered":
        return load_centered(a, rng, eeg, art["eog"], art["emg"])
    if a.artifact == "both":
        k = min(len(art["eog"]), len(art["emg"]))
        art["both"] = art["eog"][rng.permutation(len(art["eog"]))[:k]] + art["emg"][rng.permutation(len(art["emg"]))[:k]]
    idx = rng.permutation(len(eeg))
    n_va = a.n_train // 5
    sets = {}
    for name, sl in [("tr", idx[:a.n_train - n_va]), ("va", idx[a.n_train - n_va:a.n_train]),
                     ("te", idx[a.n_train:a.n_train + a.n_test])]:
        snr = rng.uniform(-7, 2, size=len(sl))
        sets[name] = contaminate(eeg[sl], art[a.artifact], snr, rng)
    return {k: v[0] for k, v in sets.items()}, {k: v[1] for k, v in sets.items()}, None


def metrics(xhat, x, y, snr0=None):
    """Per-epoch RRMSE, CC and whole-epoch SNR gain. With `snr0` (nominal input SNR per epoch,
    centered protocol) also the SPAR-EEG artifact-region delta SNR: SNR inside the middle third
    after denoising minus the nominal input SNR."""
    err = np.sqrt(((xhat - x) ** 2).mean(1))
    rrmse = err / np.sqrt((x ** 2).mean(1))
    cc = np.array([np.corrcoef(a, b)[0, 1] for a, b in zip(xhat, x)])
    snr_in = 20 * np.log10(np.sqrt((x ** 2).mean(1)) / np.sqrt(((y - x) ** 2).mean(1)))
    snr_out = 20 * np.log10(np.sqrt((x ** 2).mean(1)) / err)
    m = dict(rrmse=rrmse, cc=cc, snr_gain=snr_out - snr_in)
    if snr0 is not None:
        T = x.shape[1]
        r = slice(T // 3, 2 * (T // 3))
        m["dsnr_region"] = 10 * np.log10(x[:, r].var(1) / (xhat - x)[:, r].var(1)) - snr0
        m["snr0"] = snr0
    return m


def fmt(m):
    s = f"RRMSE {m['rrmse'].mean():.3f}  CC {m['cc'].mean():.3f}  SNR gain {m['snr_gain'].mean():+.2f} dB"
    if "dsnr_region" in m:
        by_level = [np.median(m["dsnr_region"][m["snr0"] == l]) for l in np.unique(m["snr0"])]
        s += f"  region dSNR {np.mean(by_level):+.2f} dB (mean of per-level medians)"
    return s


def levels_table(res):
    """Per-level median artifact-region delta SNR, the SPAR-EEG Fig. 3 / Table II summary."""
    names = [k for k in res if "dsnr_region" in res[k] and k != "noisy"]
    print("input SNR  " + "  ".join(f"{k[:10]:>10s}" for k in names))
    for l in LEVELS:
        print(f"{l:+6d} dB  " + "  ".join(f"{np.median(res[k]['dsnr_region'][res[k]['snr0'] == l]):+10.2f}" for k in names))
    print("     mean  " + "  ".join(f"{np.mean([np.median(res[k]['dsnr_region'][res[k]['snr0'] == l]) for l in LEVELS]):+10.2f}" for k in names))


class Ridge:
    """Streaming ridge on standardised features. Only Gram statistics are kept (train and val),
    so the feature matrices are never stored; lambda is picked by the closed-form validation error."""
    def __init__(self, n_feat, device):
        z = lambda *shape: torch.zeros(*shape, device=device, dtype=torch.float64)
        self.n = {"tr": 0, "va": 0}
        self.S1, self.S2 = {k: z(n_feat) for k in self.n}, {k: z(n_feat, n_feat) for k in self.n}
        self.SXY, self.SY, self.SYY = {k: z(n_feat) for k in self.n}, {k: 0.0 for k in self.n}, {k: 0.0 for k in self.n}

    def add(self, split, X, Y):
        """X (n, F), Y (n,) on the device."""
        X, Y = X.double(), Y.double()
        self.S1[split] += X.sum(0)
        self.S2[split] += X.T @ X
        self.SXY[split] += X.T @ Y
        self.SY[split] += Y.sum().item()
        self.SYY[split] += (Y ** 2).sum().item()
        self.n[split] += len(X)

    def _centred(self, split):
        """Gram, cross-moment and target energy of split, centred with the TRAIN mean and scale."""
        n, mu, sd, yb = self.n[split], self.mu, self.sd, self.ybar
        S1, S2, SXY, SY, SYY = self.S1[split], self.S2[split], self.SXY[split], self.SY[split], self.SYY[split]
        G = (S2 - torch.outer(S1, mu) - torch.outer(mu, S1) + n * torch.outer(mu, mu)) / torch.outer(sd, sd)
        R = (SXY - mu * SY - S1 * yb + n * mu * yb) / sd
        return G, R, SYY - 2 * yb * SY + n * yb ** 2

    def fit(self, lams=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1)):
        n = self.n["tr"]
        self.mu = self.S1["tr"] / n
        self.sd = (self.S2["tr"].diagonal() / n - self.mu ** 2).clamp_min(0).sqrt() + 1e-6
        self.ybar = self.SY["tr"] / n
        G, R, _ = self._centred("tr")
        Gv, Rv, yv2 = self._centred("va")
        eye = torch.eye(len(G), device=G.device, dtype=G.dtype)
        best = (np.inf, None, None)
        for lam in lams:
            w = torch.linalg.solve(G + lam * n * eye, R)
            e = (w @ Gv @ w - 2 * w @ Rv + yv2).item() / self.n["va"]
            if e < best[0]:
                best = (e, w, lam)
        self.val_mse, self.w, self.lam = best
        return self

    def predict(self, X):
        return ((X.double() - self.mu) / self.sd) @ self.w + self.ybar


def fir_features(y, device, taps=65):
    """Tapped delay line of the noisy signal: the 'no brain' linear-filter control."""
    h = taps // 2
    yp = np.pad(y, ((0, 0), (h, h)), mode="reflect")
    return torch.as_tensor(np.stack([yp[:, k:k + y.shape[1]] for k in range(taps)], -1), device=device)  # (B, T, taps)


def evaluate(name, feats, x, y, a, snr0=None):
    """feats(y_batch) -> (B, T, F) features on the device. Streams train and val through the ridge,
    then predicts the test set batch by batch; returns (xhat_test, per-epoch test metrics)."""
    n_feat = feats(y["tr"][:1]).shape[-1]
    dump = {k: np.lib.format.open_memmap(a.save_states / f"{k}_states.npy", "w+", np.float16, (len(y[k]), y[k].shape[1], n_feat))
            for k in y} if a.save_states else {}
    r = Ridge(n_feat, a.device)
    for split in ("tr", "va"):
        for b in range(0, len(y[split]), a.batch):
            X = feats(y[split][b:b + a.batch])
            if dump:
                dump[split][b:b + a.batch] = X.half().cpu().numpy()
            r.add(split, X.reshape(-1, n_feat), torch.as_tensor(x[split][b:b + a.batch], device=a.device).reshape(-1))
    r.fit()
    xhat = []
    for b in range(0, len(y["te"]), a.batch):
        X = feats(y["te"][b:b + a.batch])
        if dump:
            dump["te"][b:b + a.batch] = X.half().cpu().numpy()
        xhat.append(r.predict(X.reshape(-1, n_feat)).reshape(-1, y["te"].shape[1]).float().cpu().numpy())
    xhat = np.concatenate(xhat)
    for k, d in dump.items():
        d.flush()
        np.save(a.save_states / f"{k}_clean.npy", x[k]); np.save(a.save_states / f"{k}_noisy.npy", y[k])
    if dump and snr0 is not None:
        np.save(a.save_states / "te_snr.npy", snr0)
    m = metrics(xhat, x["te"], y["te"], snr0)
    print(f"{name:30s} val RMSE {np.sqrt(r.val_mse):.3f} | test {fmt(m)}   (lambda {r.lam}, {n_feat} feats)", flush=True)
    return xhat, m, r


def paired(res, a="fly", b="fir", key="snr_gain", n_boot=5000):
    """Mean per-epoch difference a - b with a bootstrap 95% CI, and the fraction of epochs a wins."""
    d = res[a][key] - res[b][key]
    rng = np.random.default_rng(0)
    boots = np.array([d[rng.integers(len(d), size=len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"paired {key}: {a} - {b} = {d.mean():+.2f} dB  95% CI [{lo:+.2f}, {hi:+.2f}]  "
          f"{a} better on {(d > 0).mean():.0%} of {len(d)} test epochs")


# ---------------------------------------------------------------- the fly ----
def spectral_radius(conn, device, iters=150):
    """|largest eigenvalue| of the signed, input-normalised weight matrix, by power iteration."""
    pre, post, w = conn.pre.to(device), conn.post.to(device), (conn.sign * conn.weight).to(device)
    v = torch.randn(conn.n_neurons, device=device, generator=torch.Generator(device).manual_seed(0))
    v /= v.norm()
    growth = []
    for _ in range(iters):
        u = torch.zeros_like(v).index_add_(0, post, v[pre] * w)
        n = u.norm()
        growth.append(n.item())
        v = u / n
    return float(np.exp(np.mean(np.log(growth[-30:]))))


def make_model(conn, a, device):
    """Connectome RNN with the dynamics we choose: W scaled to spectral radius `rho`,
    per-neuron leaks log-uniform in [alpha_min, alpha_max], tonic bias."""
    rho1 = spectral_radius(conn, device)
    m = ConnectomeRNN(conn, alpha_init=a.alpha_max, global_scale=a.rho / rho1, h_max=a.h_max,
                      bias_init=a.bias).to(device).eval()
    if a.alpha_min < a.alpha_max:
        g = torch.Generator().manual_seed(1)
        lo, hi = np.log(a.alpha_min), np.log(a.alpha_max)
        al = torch.exp(torch.rand(conn.n_neurons, generator=g) * (hi - lo) + lo)
        m.alpha_logit.data = torch.log(al / (1 - al)).to(device)
    print(f"  rho at scale 1 = {rho1:.3f}, global_scale = {a.rho / rho1:.3f}", flush=True)
    return m


def jo_drive(y, in_idx, gain):
    """Positive half-wave of the signal to one half of the JO neurons, negative to the other
    (JO neurons are direction selective)."""
    half = len(in_idx) // 2
    return torch.cat([torch.relu(y)[:, :, None].expand(-1, -1, half),
                      torch.relu(-y)[:, :, None].expand(-1, -1, len(in_idx) - half)], -1) * gain


class Reservoir:
    """Runs the brain on a batch of noisy epochs and returns readout features (B, T, K [+1]) aligned
    so that feature t is the state `lag` samples after input t. The input is reflect-padded: `warm`
    samples in front so the network is not read while starting from rest, `lag` at the end for the
    look-ahead. Keeps running activity statistics for the log line."""
    def __init__(self, model, in_idx, out_idx, a):
        self.model, self.in_idx, self.out_idx, self.a = model, in_idx.to(a.device), out_idx.to(a.device), a
        self.n = self.act = self.sat = 0.0
        self.ever = torch.zeros(len(out_idx), dtype=torch.bool, device=a.device)
        self.grad = False

    def __call__(self, y):
        a, T = self.a, y.shape[1]
        yb = torch.as_tensor(np.pad(y, ((0, 0), (a.warm, a.lag)), mode="reflect"), device=a.device)
        with torch.set_grad_enabled(self.grad):
            h = self.model(InputDrive(self.in_idx, jo_drive(yb, self.in_idx, a.gain)), record=self.out_idx)[:, 1:]
        h = h[:, a.warm + a.lag:a.warm + a.lag + T]
        with torch.no_grad():
            self.n += h.numel(); self.act += h.sum().item(); self.sat += (h >= a.h_max - 1e-3).sum().item()
            self.ever |= h.abs().amax((0, 1)) > 1e-6
        return h if a.no_skip else torch.cat([h, torch.as_tensor(y, device=a.device)[:, :, None]], -1)

    def report(self):
        print(f"  mean act {self.act / self.n:.3f}  silent {1 - self.ever.float().mean().item():.1%}  "
              f"saturated {self.sat / self.n:.2%}", flush=True)


def tune(model, ridge, in_idx, out_idx, x, y, a):
    """Retune the brain by gradient descent on the train-split MSE. Connectivity and signs stay fixed.
    `neuron`: per-neuron leak and bias.  `edge`: also a positive gain on every synapse.
    A linear head on the readout neurons is trained jointly from zero."""
    dev = a.device
    for p in model.parameters():
        p.requires_grad_(False)
    params = [model.alpha_logit, model.bias] + ([model.log_gain] if a.train == "edge" else [])
    for p in params:
        p.requires_grad_(True)
    # a head initialised from the ridge solution is too fragile (large cancelling weights): the
    # first optimiser step on the brain breaks the cancellation. So the head starts at zero and
    # co-adapts; the reported score is a fresh ridge fitted on the tuned brain afterwards.
    mu, sd = ridge.mu.float(), ridge.sd.clamp_min(1e-2 * ridge.sd.median()).float()
    w = torch.nn.Parameter(torch.zeros(len(mu), device=dev))
    b = torch.nn.Parameter(torch.tensor(float(ridge.ybar), device=dev))
    opt = torch.optim.Adam([{"params": params, "lr": a.lr}, {"params": [w, b], "lr": 1e-3}])
    res = Reservoir(model, in_idx, out_idx, a)
    res.grad = True
    n, g, t0 = len(x["tr"]), torch.Generator().manual_seed(0), time.time()
    for ep in range(a.epochs):
        perm, tot = torch.randperm(n, generator=g).numpy(), 0.0
        for i0 in range(0, n, a.train_batch):
            i = np.sort(perm[i0:i0 + a.train_batch])
            pred = ((res(y["tr"][i]) - mu) / sd) @ w + b
            loss = ((pred - torch.as_tensor(x["tr"][i], device=dev)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params + [w, b], 1.0)
            opt.step()
            tot += loss.item() * len(i)
        print(f"  tune pass {ep + 1}/{a.epochs}: train RMSE {np.sqrt(tot / n):.3f}  {time.time() - t0:.0f}s", flush=True)
    for p in params:
        p.requires_grad_(False)
    al = model.alpha().detach()
    print(f"  after tuning: alpha median {al.median():.3f} [{al.min():.3f}, {al.max():.3f}]  bias mean {model.bias.mean():.3f}"
          + (f"  gain median {model.log_gain.exp().median():.3f} [{model.log_gain.exp().min():.2f}, {model.log_gain.exp().max():.2f}]" if a.train == "edge" else ""), flush=True)


def run_fly(conn, label, in_idx, out_idx, x, y, a, snr0=None):
    """Fixed-brain evaluation, then (if --train) tuning and a second evaluation with a fresh readout."""
    t0 = time.time()
    model = make_model(conn, a, a.device)
    res = Reservoir(model, in_idx, out_idx, a)
    xhat, m, ridge = evaluate(label, res, x, y, a, snr0)
    res.report()
    print(f"  {time.time() - t0:.0f}s", flush=True)
    if a.train == "none":
        return xhat, m
    tune(model, ridge, in_idx, out_idx, x, y, a)
    res = Reservoir(model, in_idx, out_idx, a)
    xhat, m, _ = evaluate(f"{label} TUNED ({a.train})", res, x, y, a, snr0)
    res.report()
    print(f"  {time.time() - t0:.0f}s", flush=True)
    return xhat, m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", default="eog", choices=["eog", "emg", "both"])
    p.add_argument("--protocol", default="whole", choices=["whole", "centered"],
                   help="whole: EEGdenoiseNet mixing over the full epoch, SNR U(-7, 2); centered: SPAR-EEG middle-third layout, test at 26 levels (n-test = records per level)")
    p.add_argument("--n-train", type=int, default=800)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--batch", type=int, default=40)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--subset", default="central", choices=["central", "all"])
    p.add_argument("--readout", default="dn", choices=["dn", "random"], help="dn: descending + motor; random: 2000 random non-JO neurons")
    # the brain's assumed physics (the connectome has no time constants or strengths in physical units)
    p.add_argument("--rho", type=float, default=0.9, help="spectral radius of the signed weight matrix")
    p.add_argument("--alpha-min", type=float, default=0.02, help="slowest per-step leak (memory ~ 1/alpha samples)")
    p.add_argument("--alpha-max", type=float, default=0.5)
    p.add_argument("--gain", type=float, default=1.0, help="JO drive per unit signal (signal has std 1)")
    p.add_argument("--bias", type=float, default=0.02, help="tonic drive to every neuron")
    p.add_argument("--h-max", type=float, default=10.0)
    p.add_argument("--lag", type=int, default=24, help="readout look-ahead in samples (FIR control sees +-32)")
    p.add_argument("--warm", type=int, default=64)
    p.add_argument("--no-skip", action="store_true", help="no direct noisy-input feature in the readout")
    p.add_argument("--train", default="none", choices=["none", "neuron", "edge"], help="retune the brain: per-neuron leak+bias, or also per-synapse gain")
    p.add_argument("--epochs", type=int, default=3, help="tuning passes over the train split")
    p.add_argument("--train-batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--no-fir", action="store_true")
    p.add_argument("--save-states", type=Path, default=None, help="dump the fly readout features (fp16 memmaps) to this folder for readout experiments")
    p.add_argument("--no-shuffle-control", action="store_true")
    a = p.parse_args()
    rng = np.random.default_rng(0)

    x, y, snr = load_data(a, rng)
    snr0 = snr["te"] if snr else None
    print(f"EEGdenoiseNet {a.artifact} ({a.protocol}): train {len(x['tr'])} val {len(x['va'])} test {len(x['te'])} epochs of {x['tr'].shape[1]} samples", flush=True)
    res = {"noisy": metrics(y["te"], x["te"], y["te"], snr0)}
    print(f"{'noisy input (no processing)':30s} test {fmt(res['noisy'])}")
    if not a.no_fir:
        save, a.save_states = a.save_states, None
        _, res["fir"], _ = evaluate("linear FIR (65 taps, no brain)", lambda yb: fir_features(yb, a.device), x, y, a, snr0)
        a.save_states = save

    conn = load_malecns(NFLY_DATA)
    if a.subset == "central":
        keep = conn.neurons.super_class.isin(CENTRAL) | conn.neurons.cell_type.str.startswith("JO", na=False)
        conn = conn.subset(torch.as_tensor(np.flatnonzero(keep.to_numpy())))
    ne = conn.neurons
    is_jo = ne.cell_type.str.startswith("JO", na=False).to_numpy()
    in_idx = torch.as_tensor(np.flatnonzero(is_jo))
    if a.readout == "dn":
        out_idx = torch.as_tensor(np.flatnonzero(ne.super_class.isin(["descending_neuron"]).to_numpy() | (ne.flow == "efferent").to_numpy()))
    else:
        out_idx = torch.as_tensor(np.random.default_rng(1).choice(np.flatnonzero(~is_jo), 2000, replace=False))
    print(f"{conn.summary()}\ninput: {len(in_idx)} Johnston's organ neurons; readout: {len(out_idx)} ({a.readout})", flush=True)

    if a.save_states:
        a.save_states.mkdir(parents=True, exist_ok=True)
    xhat_fly, res["fly"] = run_fly(conn, "FLY connectome reservoir", in_idx, out_idx, x, y, a, snr0)
    a.save_states = None

    # control: same neurons, same edge count, same signs, same spectral radius, wiring destroyed
    if not a.no_shuffle_control:
        g = torch.Generator().manual_seed(0)
        shuffled = build_connectome(conn.neurons, conn.pre, conn.post[torch.randperm(conn.n_edges, generator=g)], conn.syn_count, conn.sign)
        _, res["shuffled"] = run_fly(shuffled, "shuffled-wiring reservoir", in_idx, out_idx, x, y, a, snr0)

    for other in ("fir", "shuffled"):
        if other in res:
            paired(res, "fly", other)
    if snr0 is not None:
        levels_table(res)
    np.savez(HERE / f"results_{a.artifact}{'' if a.protocol == 'whole' else '_centered'}.npz", clean=x["te"], noisy=y["te"], fly=xhat_fly, args=str(vars(a)),
             **{f"{m}_{k}": v for m, r in res.items() for k, v in r.items()})
    print("saved", HERE / f"results_{a.artifact}{'' if a.protocol == 'whole' else '_centered'}.npz")


if __name__ == "__main__":
    main()
