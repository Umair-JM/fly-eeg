"""Readout experiments on cached fly features (from fly_eeg_denoise.py --save-states DIR).
The brain is fixed; only what listens to the descending / motor neurons changes:
  ridge          one weight per neuron (the baseline readout, should reproduce the main script)
  ridge + shifts the same neurons read at several time offsets
  mlp            a small two-layer network on the (shifted) neuron activities

Run from the nfly folder:  uv run --no-sync python "../fly-eeg/fly_eeg_readout.py" DIR
"""
import argparse, time
from pathlib import Path
import numpy as np, torch
from torch import nn
from fly_eeg_denoise import metrics, fmt, paired, Ridge, levels_table


def load(d):
    S = {k: np.load(d / f"{k}_states.npy", mmap_mode="r") for k in ("tr", "va", "te")}
    x = {k: np.load(d / f"{k}_clean.npy") for k in S}
    y = {k: np.load(d / f"{k}_noisy.npy") for k in S}
    snr0 = np.load(d / "te_snr.npy") if (d / "te_snr.npy").exists() else None
    return S, x, y, snr0


def shifted(X, shifts):
    """X (B, T, F) -> (B, T, F * len(shifts)): X[:, t + d] (edge clamped) for each d."""
    T = X.shape[1]
    idx = torch.arange(T, device=X.device)
    return torch.cat([X[:, (idx + d).clamp(0, T - 1)] for d in shifts], -1)


def batches(S, split, dev, n_shifts=1):
    batch = max(4, 25 // n_shifts)      # keeps the (rows x features) block on the GPU roughly constant
    for b in range(0, len(S[split]), batch):
        yield b, torch.as_tensor(np.array(S[split][b:b + batch]), device=dev).float()


def ridge_eval(name, S, x, y, shifts, dev, snr0=None):
    F = S["tr"].shape[-1] * len(shifts)
    r = Ridge(F, dev)
    for split in ("tr", "va"):
        for b, X in batches(S, split, dev, len(shifts)):
            r.add(split, shifted(X, shifts).reshape(-1, F), torch.as_tensor(x[split][b:b + len(X)], device=dev).reshape(-1))
    r.fit()
    xhat = np.concatenate([r.predict(shifted(X, shifts).reshape(-1, F)).reshape(len(X), -1).float().cpu().numpy()
                           for _, X in batches(S, "te", dev, len(shifts))])
    m = metrics(xhat, x["te"], y["te"], snr0)
    print(f"{name:34s} val RMSE {np.sqrt(r.val_mse):.3f} | test {fmt(m)}   (lambda {r.lam}, {F} feats)", flush=True)
    return xhat, m


class MLP(nn.Module):
    def __init__(self, mu, sd, F, hidden):
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("sd", sd)
        self.net = nn.Sequential(nn.Linear(F, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, X):
        return self.net(((X - self.mu) / self.sd).clamp(-10, 10)).squeeze(-1)


def mlp_eval(name, S, x, y, shifts, dev, hidden=512, steps=6000, batch=1024, lr=1e-3, seed=0, snr0=None):
    """Two hidden layers on the standardised (shifted) activities; Adam, cosine schedule, best-on-val."""
    torch.manual_seed(seed)
    K = S["tr"].shape[-1]
    Str = S["tr"]                                                                # fp16 memmap (N, T, K), never copied
    xtr = torch.as_tensor(x["tr"])
    mu = torch.stack([X.reshape(-1, K).mean(0) for _, X in batches(S, "tr", dev)]).mean(0)
    sd = torch.stack([X.reshape(-1, K).var(0) for _, X in batches(S, "tr", dev)]).mean(0).sqrt()
    sd = sd.clamp_min(1e-2 * sd.median())
    model = MLP(mu.repeat(len(shifts)), sd.repeat(len(shifts)), K * len(shifts), hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    N, T = xtr.shape
    g = torch.Generator().manual_seed(seed)

    def predict(split):
        with torch.no_grad():
            return np.concatenate([model(shifted(X, shifts)).cpu().numpy() for _, X in batches(S, split, dev, len(shifts))])

    best, t0 = (np.inf, None), time.time()
    for step in range(1, steps + 1):
        e = torch.randint(N, (batch,), generator=g).numpy()
        t = torch.randint(T, (batch,), generator=g).numpy()
        X = torch.as_tensor(np.concatenate([Str[e, np.clip(t + d, 0, T - 1)] for d in shifts], -1), device=dev).float()
        loss = ((model(X) - xtr[e, t].to(dev)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0:
            v = np.sqrt(((predict("va") - x["va"]) ** 2).mean())
            if v < best[0]:
                best = (v, {k: t_.clone() for k, t_ in model.state_dict().items()})
            print(f"  step {step}: train RMSE {loss.sqrt().item():.3f}  val RMSE {v:.3f}  {time.time() - t0:.0f}s", flush=True)
    model.load_state_dict(best[1])
    xhat = predict("te")
    m = metrics(xhat, x["te"], y["te"], snr0)
    print(f"{name:34s} val RMSE {best[0]:.3f} | test {fmt(m)}   ({sum(p.numel() for p in model.parameters()):,} params)", flush=True)
    return xhat, m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir", type=Path)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--shifts", default="-24,-12,0,12,24", help="time offsets (samples) at which the readout neurons are read")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--steps", type=int, default=6000)
    a = p.parse_args()
    S, x, y, snr0 = load(a.dir)
    shifts = [int(s) for s in a.shifts.split(",")]
    print(f"states {S['tr'].shape} train, {S['va'].shape} val, {S['te'].shape} test", flush=True)
    res = {"noisy": metrics(y["te"], x["te"], y["te"], snr0)}
    _, res["ridge"] = ridge_eval("ridge, 1 shift (baseline readout)", S, x, y, [0], a.device, snr0)
    _, res["ridge_shifts"] = ridge_eval(f"ridge, {len(shifts)} shifts", S, x, y, shifts, a.device, snr0)
    _, res["mlp"] = mlp_eval("mlp, 1 shift", S, x, y, [0], a.device, a.hidden, a.steps, snr0=snr0)
    xhat, res["mlp_shifts"] = mlp_eval(f"mlp, {len(shifts)} shifts", S, x, y, shifts, a.device, a.hidden, a.steps, snr0=snr0)
    for k in ("ridge_shifts", "mlp", "mlp_shifts"):
        paired(res, k, "ridge")
    if snr0 is not None:
        levels_table(res)
    np.savez(a.dir / "readouts.npz", clean=x["te"], noisy=y["te"], mlp_shifts=xhat,
             **{f"{m}_{k}": v for m, r in res.items() for k, v in r.items()})


if __name__ == "__main__":
    main()
