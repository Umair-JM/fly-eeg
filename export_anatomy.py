"""Fetch official MaleCNS skeletons for the neurons the simulation draws and pack them, with the brain
neuropil shell, into sim/data: skel.bin (float32 segment endpoints), skel_id.bin (uint32 neuron
slot per vertex), shell.bin (float32 vertices + uint32 faces of the brain neuropils) and
anatomy.json (slot -> connectome node index, group, body id).

Skeleton source: v1.0/segmentation/skeletons-malecns/skeletons-swc/<body>.swc (public bucket, 8 nm
voxels). Neuropil meshes: nfly data/anatomy/neuropils.json (scripts/fetch_anatomy.py).
Run from the nfly folder after export_sim.py:  uv run --no-sync python ../fly-eeg/export_anatomy.py
"""
import argparse, base64, json, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, torch
import fly_eeg_denoise as F
from nfly.viz.anatomy import read_swc

BUCKET = "https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc"
SKEL_DIR = F.NFLY_DATA / "anatomy" / "skeletons"


def fetch_swc(body):
    p = SKEL_DIR / f"{body}.swc"
    if not p.exists():
        try:
            p.write_bytes(urllib.request.urlopen(f"{BUCKET}/{body}.swc", timeout=60).read())
        except Exception as e:
            print("no skeleton for", body, e)
            return None
    return p


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=F.HERE / "sim" / "data")
    p.add_argument("--n-targets", type=int, default=250, help="first synaptic targets of the antenna neurons")
    p.add_argument("--n-brain", type=int, default=400, help="random central brain neurons for context")
    p.add_argument("--max-nodes", type=int, default=110)
    a = p.parse_args()
    rng = np.random.default_rng(0)

    conn = F.load_malecns(F.NFLY_DATA)
    ne = conn.neurons
    keep = ne.super_class.isin(F.CENTRAL) | ne.cell_type.str.startswith("JO", na=False)
    conn = conn.subset(torch.as_tensor(np.flatnonzero(keep.to_numpy())))
    ne = conn.neurons
    is_jo = ne.cell_type.str.startswith("JO", na=False).to_numpy()
    is_ro = (ne.super_class.isin(["descending_neuron"]) | (ne.flow == "efferent")).to_numpy()
    pre, post, syn = conn.pre.numpy(), conn.post.numpy(), conn.syn_count.numpy()
    jo_in = np.zeros(len(ne)); np.add.at(jo_in, post[is_jo[pre]], syn[is_jo[pre]])
    tgt = np.flatnonzero((jo_in > 0) & ~is_jo & ~is_ro)
    tgt = rng.choice(tgt, min(a.n_targets, len(tgt)), replace=False, p=jo_in[tgt] / jo_in[tgt].sum())
    rest = np.flatnonzero(~is_jo & ~is_ro & (jo_in == 0))
    brain = rng.choice(rest, a.n_brain, replace=False)
    half = is_jo.sum() // 2
    jo = np.flatnonzero(is_jo)
    slots = [(i, 1) for i in jo[:half]] + [(i, 2) for i in jo[half:]] + [(i, 0) for i in tgt] + [(i, 0) for i in brain] + [(i, 3) for i in np.flatnonzero(is_ro)]
    print(f"{len(slots)} neurons: {is_jo.sum()} antenna, {len(tgt)} first targets, {len(brain)} brain, {is_ro.sum()} descending/motor", flush=True)

    SKEL_DIR.mkdir(parents=True, exist_ok=True)
    bodies = [int(ne.root_id.iloc[i]) for i, _ in slots]
    with ThreadPoolExecutor(16) as ex:
        paths = list(ex.map(fetch_swc, bodies))
    seg, sid, meta = [], [], []
    for (node, g), body, path in zip(slots, bodies, paths):
        if path is None:
            continue
        xyz, parent = read_swc(path, a.max_nodes)
        e = np.flatnonzero(parent >= 0)
        seg.append(np.stack([xyz[parent[e]], xyz[e]], 1).reshape(-1, 3))
        sid.append(np.full(2 * len(e), len(meta), np.uint32))
        meta.append(dict(node=int(node), group=int(g), body=body))
    seg, sid = np.concatenate(seg).astype(np.float32), np.concatenate(sid)
    seg.tofile(a.out / "skel.bin"); sid.tofile(a.out / "skel_id.bin")
    print(f"{len(meta)} skeletons, {len(seg) // 2:,} segments, {seg.nbytes / 1e6:.0f} MB", flush=True)

    # brain neuropils: one mesh each, so a region can glow on its own
    rois = [r for r in json.loads((F.NFLY_DATA / "anatomy" / "neuropils.json").read_text())["rois"] if r.get("region") != "vnc"]
    V, Fc, regions, off, foff = [], [], [], 0, 0
    for k, r in enumerate(rois):
        v = np.frombuffer(base64.b64decode(r["vertices"]), np.float32).reshape(-1, 3)
        f = np.frombuffer(base64.b64decode(r["faces"]), np.uint32).reshape(-1, 3)
        regions.append(dict(name=r["name"], label=r["name"].split("(")[0], v0=off, nv=len(v), f0=foff, nf=len(f), centre=v.mean(0).round(1).tolist()))
        V.append(v); Fc.append(f + off); off += len(v); foff += len(f)
    V, Fc = np.concatenate(V), np.concatenate(Fc)
    with open(a.out / "shell.bin", "wb") as fh:
        fh.write(np.array([len(V), len(Fc)], np.uint32).tobytes()); fh.write(V.astype(np.float32).tobytes()); fh.write(Fc.astype(np.uint32).tobytes())
    print(f"shell: {len(V):,} vertices, {len(Fc):,} faces from {len(rois)} brain neuropils", flush=True)

    # which regions each neuron runs through: every skeleton node -> nearest neuropil mesh vertex
    # (a coarse grid nearest-neighbour search, no scipy in this environment)
    cell, step = 8.0, 8                                      # grid size in um; every 8th skeleton node
    region_of_vertex = np.concatenate([np.full(r["nv"], k, np.int32) for k, r in enumerate(regions)])
    vkeys = np.floor(V / cell).astype(np.int64)
    grid = {}
    for i, key in enumerate(map(tuple, vkeys)):
        grid.setdefault(key, []).append(i)
    grid = {k: np.array(v) for k, v in grid.items()}
    nodes = seg.reshape(-1, 2, 3)[::step, 1]
    node_slot = sid.reshape(-1, 2)[::step, 1]
    groups = {}
    for i, key in enumerate(map(tuple, np.floor(nodes / cell).astype(np.int64))):
        groups.setdefault(key, []).append(i)
    near = np.full(len(nodes), -1, np.int32)
    shifts = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    for key, idx in groups.items():
        cand = [grid[k2] for k2 in ((key[0] + dx, key[1] + dy, key[2] + dz) for dx, dy, dz in shifts) if k2 in grid]
        if not cand:
            continue
        cand = np.concatenate(cand)
        d2 = ((nodes[idx][:, None, :] - V[cand][None, :, :]) ** 2).sum(-1)
        j = d2.argmin(1)
        near[idx] = np.where(d2[np.arange(len(idx)), j] < cell ** 2, region_of_vertex[cand[j]], -1)
    counts = np.zeros((len(meta), len(regions)), np.float32)
    ok = near >= 0
    np.add.at(counts, (node_slot[ok], near[ok]), 1)
    for m, row in zip(meta, counts):
        tot = row.sum()
        top = np.argsort(-row)[:4]
        m["regions"] = {int(k): round(float(row[k] / tot), 3) for k in top if row[k] > 0} if tot else {}
    inside = ok.mean()
    print(f"{inside:.0%} of skeleton nodes lie within 20 um of a neuropil", flush=True)

    # the signal's route, from the synapses: hop distance of every neuron from the antenna, then the
    # synapse flow between the main regions of the drawn neurons along hop k -> hop k+1 edges
    hop = np.full(len(ne), -1, np.int32); hop[is_jo] = 0
    frontier = is_jo.copy()
    for k in range(1, 6):
        nxt = np.zeros(len(ne), bool); nxt[post[frontier[pre]]] = True; nxt &= hop < 0
        hop[nxt] = k; frontier = nxt
    skip = {"ME", "LO", "LOP", "LA", "AME", "CV-anterior", "CRN"}
    top_region = np.full(len(ne), -1, np.int32)
    for m, row in zip(meta, counts):
        row = row.copy()
        for k, r in enumerate(regions):
            if r["label"] in skip: row[k] = 0
        if row.sum(): top_region[m["node"]] = int(row.argmax())
        m["hop"] = int(hop[m["node"]])
    src = np.where(is_jo[pre], -1, top_region[pre]); dst = top_region[post]
    use = (dst >= 0) & ((src >= 0) | is_jo[pre]) & (hop[pre] >= 0) & (hop[post] == hop[pre] + 1) & (src != dst)
    flows = {}
    for s_, d_, w, h in zip(src[use], dst[use], syn[use], hop[pre][use]):
        key = (int(s_), int(d_), int(h)); flows[key] = flows.get(key, 0.0) + float(w)
    flows = sorted(flows.items(), key=lambda kv: -kv[1])
    keep_flows = []
    for (s_, d_, h), w in flows:
        if h <= 3 and (len([f for f in keep_flows if f["hop"] == h]) < 8):
            keep_flows.append(dict(src=s_, dst=d_, hop=h, w=round(w)))
    dn_regions = {}
    for m in meta:
        if m["group"] == 3 and top_region[m["node"]] >= 0: dn_regions[int(top_region[m["node"]])] = dn_regions.get(int(top_region[m["node"]]), 0) + 1
    print("route:", [(regions[f["src"]]["label"] if f["src"] >= 0 else "Antenna", regions[f["dst"]]["label"], f["hop"], f["w"]) for f in keep_flows[:12]])
    print("descending neurons mostly in:", sorted(dn_regions.items(), key=lambda kv: -kv[1])[:5])
    (a.out / "anatomy.json").write_text(json.dumps(dict(neurons=meta, regions=regions, flows=keep_flows,
                                                         dn_regions=sorted(dn_regions.items(), key=lambda kv: -kv[1])[:6])))


if __name__ == "__main__":
    main()
