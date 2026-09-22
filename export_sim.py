"""Export a browser simulation of the fly EEG denoiser: the central-brain connectome is run on a few
real EEGdenoiseNet test epochs and the activity of every neuron is recorded at every sample, with
the neurons' real soma positions, the noisy input, the clean target and the decoded output.

Output folder (default sim/data): pos.bin (float32 N x 3, micrometres), group.bin
(uint8 N: 0 brain, 1 antenna +, 2 antenna -, 3 readout, 255 no position), act_<k>.bin (uint8 T x N
per epoch, sqrt-compressed activity), meta.json (traces and metrics per epoch).

Needs a feature cache with readouts: fly_eeg_denoise.py --save-states DIR, then fly_eeg_readout.py DIR.
Run from the nfly folder:  uv run --no-sync python ../fly-eeg/export_sim.py --cache DIR
Then serve fly-eeg/sim (python -m http.server) and open index.html.
"""
import argparse, json
from pathlib import Path
import numpy as np, torch
import fly_eeg_denoise as F

def pick_epochs(x, y, decoded, targets=(-6.0, -2.0, 1.5)):
    """One epoch near each target input SNR, with decoded quality near the median so the picks are typical."""
    m = F.metrics(decoded, x, y)
    snr_in = 20 * np.log10(np.sqrt((x ** 2).mean(1)) / np.sqrt(((y - x) ** 2).mean(1)))
    med = np.median(m["snr_gain"])
    return [int(np.argmin(np.abs(snr_in - t) + 0.3 * np.abs(m["snr_gain"] - med))) for t in targets], m, snr_in


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True, help="folder written by --save-states and fly_eeg_readout.py")
    p.add_argument("--out", type=Path, default=F.HERE / "sim" / "data")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for k, v in dict(rho=0.9, alpha_min=0.02, alpha_max=0.5, gain=1.0, bias=0.02, h_max=10.0, lag=24, warm=64).items():
        setattr(a, k, v)

    x, y = np.load(a.cache / "te_clean.npy"), np.load(a.cache / "te_noisy.npy")
    decoded = np.load(a.cache / "readouts.npz")["mlp_shifts"]
    picks, m, snr_in = pick_epochs(x, y, decoded)
    print("epochs", picks, "input SNR", snr_in[picks].round(1), "gain", m["snr_gain"][picks].round(1))

    conn = F.load_malecns(F.NFLY_DATA)
    ne = conn.neurons
    keep = ne.super_class.isin(F.CENTRAL) | ne.cell_type.str.startswith("JO", na=False)
    conn = conn.subset(torch.as_tensor(np.flatnonzero(keep.to_numpy())))
    ne = conn.neurons
    is_jo = ne.cell_type.str.startswith("JO", na=False).to_numpy()
    in_idx = torch.as_tensor(np.flatnonzero(is_jo))
    is_ro = (ne.super_class.isin(["descending_neuron"]) | (ne.flow == "efferent")).to_numpy()

    # positions: real somata; the antenna neurons get two clusters beyond the left / right AMMC targets
    pos = ne[["x", "y", "z"]].to_numpy().astype(np.float32)
    has = np.isfinite(pos).all(1)
    centre = np.nanmean(pos[has], 0)
    tgt = np.zeros(len(ne), bool)
    tgt[conn.post[is_jo[conn.pre.numpy()]].numpy()] = True
    group = np.where(is_ro, 3, 0).astype(np.uint8)
    half = len(in_idx) // 2
    rng = np.random.default_rng(0)
    for g, sl, side in ((1, in_idx[:half], "L"), (2, in_idx[half:], "R")):
        c = np.nanmean(pos[tgt & has & (ne.side == side).to_numpy()], 0)
        antenna = centre + 1.6 * (c - centre)
        pos[sl.numpy()] = antenna + rng.normal(0, 12, (len(sl), 3)).astype(np.float32)
        group[sl.numpy()] = g
    group[~np.isfinite(pos).all(1)] = 255
    pos[~np.isfinite(pos).all(1)] = centre
    pos.tofile(a.out / "pos.bin")
    group.tofile(a.out / "group.bin")

    model = F.make_model(conn, a, a.device)
    T = x.shape[1]
    yp = np.pad(y[picks], ((0, 0), (a.warm, a.lag)), mode="reflect")
    with torch.no_grad():
        h = model(F.InputDrive(in_idx.to(a.device), F.jo_drive(torch.as_tensor(yp, device=a.device), in_idx.to(a.device), a.gain)))[:, 1:]
    h = h[:, a.warm:a.warm + T].cpu().numpy()                       # (E, T, N), state at the time of each input sample
    ref = float(np.quantile(h[:, :, ~is_jo], 0.995))
    act = (np.sqrt(np.clip(h / ref, 0, 1)) * 255).astype(np.uint8)
    for k in range(len(picks)):
        act[k].tofile(a.out / f"act_{k}.bin")
    print(f"activity reference {ref:.3f}; wrote {act.nbytes / 1e6:.0f} MB of activity for {len(ne)} neurons")

    meta = dict(n=int(len(ne)), t=int(T), fs=F.FS, lag=a.lag, n_readout=int(is_ro.sum()), n_antenna=int(is_jo.sum()),
                epochs=[dict(id=int(i), snr_in=float(snr_in[i]), snr_gain=float(m["snr_gain"][i]), cc=float(m["cc"][i]),
                             rrmse=float(m["rrmse"][i]), noisy=y[i].round(4).tolist(), clean=x[i].round(4).tolist(),
                             decoded=decoded[i].round(4).tolist()) for i in picks])
    (a.out / "meta.json").write_text(json.dumps(meta))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
