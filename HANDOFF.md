# Fly brain EEG denoiser: handoff to NeSI

Written 2026-09-22 from the laptop session (Windows, RTX 4050 6 GB). Read this first, then the two
scripts. Everything below was measured; nothing is projected.

## The idea in one paragraph

The MaleCNS fly connectome (Janelia v1.0, 166,700 neurons) is used as a fixed recurrent network.
Noisy single channel EEG is fed into the 672 Johnston's organ neurons (the antennal vibration
sensors), the network runs one step per EEG sample, and the clean EEG is decoded from the activity
of the 1,421 descending and motor neurons. The wiring, synapse counts and transmitter signs come
from the connectome and are never changed. What the connectome does not contain (time constants,
global gain, tonic bias) is set by hand; what listens to the output neurons (the readout) is fitted.
Umair's stated goal: a good EEG cleaner. He is fine with retuning the brain (training leaks, biases
and per synapse gains, wiring and signs fixed) if that helps.

## Files

- `fly_eeg_denoise.py`  data generation, the brain, ridge readout, controls, optional feature cache,
  optional gradient tuning (`--train neuron|edge`, works but ill conditioned, see below).
- `fly_eeg_readout.py`  readout experiments on a cached feature folder: ridge, ridge over several
  time offsets, MLP. Run after `fly_eeg_denoise.py --save-states DIR`.
- `results_*.npz`       per epoch metrics of the laptop runs (keys `<method>_<metric>`).
- `data/`               EEGdenoiseNet arrays; the script downloads them if missing (53 MB).

Both scripts import `nfly` (https://github.com/zhengxuyu/nfly.git, commit 01cf11d or later) and are
run from the nfly folder:

```
git clone https://github.com/zhengxuyu/nfly.git && cd nfly && uv sync
B=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
curl -o data/body-annotations.feather        $B/body-annotations-male-cns-v1.0-minconf-0.5.feather
curl -o data/body-neurotransmitters.feather  $B/body-neurotransmitters-male-cns-v1.0.feather
curl -o data/connectome-weights.feather      $B/connectome-weights-male-cns-v1.0-minconf-0.5.feather
uv run --no-sync python ../fly-eeg/fly_eeg_denoise.py --artifact eog --n-train 4000 --n-test 500
```

The first load parses the 1 GB weights table and caches `data/cache/malecns_min3_v2.pt` (323 MB).
Copying that cache file from the laptop skips the download. The scripts expect `fly-eeg/` and
`nfly/` to be sibling folders.

## What made it work (all in the script defaults)

1. Central brain subset: `cb_intrinsic` + descending + `cb_motor` + JO, 34,257 neurons, 2.87M
   edges. Optic lobes and VNC dropped.
2. Weight matrix scaled to spectral radius 0.9 (`--rho`). The first attempt ran the whole brain at
   an effective radius near 2 and was self driven; the EEG barely moved it.
3. Per neuron leaks log uniform in [0.02, 0.5] per sample (`--alpha-min/max`). One shared leak
   collapses performance (+5.6 dB instead of +9).
4. Tonic bias 0.02 (`--bias`): no silent neurons, inhibition becomes readable.
5. Readout look ahead of 24 samples (`--lag`), like a centred FIR. Causal readout loses 2 dB.
6. Reflect padded warm up so the network is not read while starting from rest.
7. Ridge lambda grid down to 1e-5; the ridge is streamed (Gram matrices only), so RAM is flat.

Spectral radius, input gain and which neurons are read barely matter. The network is close to
linear at this operating point; the leaks provide the multi timescale memory.

## Results (EOG, EEGdenoiseNet protocol, SNR uniform in -7 to 2 dB, same 500 test epochs)

| Cleaner | RRMSE | CC | SNR gain |
|---|---|---|---|
| noisy input | 1.417 | 0.592 | 0 dB |
| linear FIR, 65 taps, no brain | 0.543 | 0.838 | +8.09 dB |
| fly, ridge readout | 0.431 | 0.899 | +10.21 dB |
| fly, ridge over 5 time offsets (-24..24) | 0.401 | 0.912 | +10.90 dB |
| fly, MLP (512x512) over 5 offsets, 6k steps | 0.382 | 0.920 | +11.33 dB |

Paired fly minus FIR: +2.32 dB, 95% CI [2.13, 2.52], fly better on 84% of epochs
(800/200/300 run: +2.62 dB, CI [2.39, 2.85]).

The MLP trained for 20k steps overfits (train 0.180 vs val 0.250, test +11.07 dB). Longer training
is not the lever; more training data or regularisation is.

## The uncomfortable control

Shuffled wiring (post synaptic targets permuted, same neurons, edge count, signs and spectral
radius) scores the same as the real wiring once training data is plentiful: +10.42 vs +10.38 dB,
paired difference -0.04 dB, CI [-0.12, +0.05], on 300 test epochs. An earlier advantage of the real
wiring on a 150 epoch split was a small sample effect. Any claim about the connectome itself must
survive this control; a claim about "a connectome scale, Dale constrained fixed recurrent network"
does.

## Comparison with SPAR-EEG (Shaikh, Kalra, Lowe, Niazi, TNSRE 2026, DOI 10.1109/TNSRE.2026.3734253)

His protocol differs from EEGdenoiseNet's: records paired by index, artifact only in the middle
third of the 2 s epoch, clean EEG scaled to unit variance inside that window, SNR defined inside
it, 26 integer levels from -20 to +5 dB, score = median artifact region SNR improvement per level,
then the mean over levels. His Table II: SPAR-EEG 9.05 / 8.28 / 8.00 dB (EMG / EOG / EOG+EMG),
WQN 9.01 / 6.67 / 7.11, wavelet hard 6.60 / 2.24 / 3.38, EMD-CCA 3.92 / 3.25 / 1.82. He tested no
neural denoisers. His generation script is public (github.com/usmanqamarshaikh/SPAR-EEG,
benchmarks/ground_truth/scripts/prepare_datasets_centered.py) and is reproduced exactly in
`load_centered()`; `--protocol centered` selects it, `--n-test` is then records per level, and the
per level table is printed at the end. Smoke tested only. The comparable run is:

```
uv run --no-sync python ../fly-eeg/fly_eeg_denoise.py --protocol centered --artifact eog \
    --n-train 2400 --n-test 150 --no-shuffle-control --save-states /path/states_eog_centered
uv run --no-sync python ../fly-eeg/fly_eeg_readout.py /path/states_eog_centered --steps 6000
```

Then the same for `--artifact emg` and `--artifact both`.

## Gradient tuning of the brain (what was tried, what broke)

`--train neuron` trains per neuron leak and bias; `--train edge` also a positive gain per synapse
(2.87M parameters). BPTT through ~600 steps at batch 8 uses about 4 GB and costs ~1 s per epoch on
the 4050. Initialising the head from the ridge solution fails: the ridge head has large cancelling
weights over correlated neurons and one Adam step on the brain breaks the cancellation (loss goes
from 0.08 to hundreds at any learning rate, because Adam moves every parameter by the full rate).
A head trained from zero jointly does not blow up but converged slowly in the few passes tried.
Untested ideas, in order: larger batch on a big GPU with SGD plus clipping instead of Adam;
per cell type gains instead of per synapse; training the head alone first with the brain fixed,
then unfreezing the brain with a much smaller rate.

## Costs measured on the laptop

- fixed brain, central subset: 0.34 s per epoch (batch 50); full EEGdenoiseNet pass 26 min
- feature cache: 512 x 1422 fp16 per epoch, 1.46 MB per epoch (4500 epochs = 6.5 GB)
- ridge on cache: under a minute; MLP 6k steps: 1 to 2 min
- BPTT tuning: ~1 s per training epoch at batch 8, 4 GB GPU

## Sensible next steps

1. Usman protocol run for EOG, EMG, EOG+EMG (the head to head he cannot yet make).
2. Data augmentation: several artifact draws and SNRs per clean epoch; the MLP was data limited.
3. A single readout trained on all three artifact types (what a deployed cleaner needs).
4. Brain tuning on an A100 with the ordering above; report fixed vs tuned vs tuned shuffled.
5. Real recordings without ground truth (his PhysioBank motion set, the RSVP/P300 speller data)
   scored by downstream task accuracy.
