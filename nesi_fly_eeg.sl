#!/bin/bash -e
# Slurm template for NeSI. Fill in the account and check the GPU spec and module names for the
# cluster you are on (sinfo / module spider CUDA); nothing below has been run on NeSI yet.
#SBATCH --job-name=fly-eeg
#SBATCH --account=[NEED: NeSI project code]
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=A100:1
#SBATCH --output=fly-eeg-%j.log

module purge
module load CUDA               # [NEED: exact module version on the cluster]

# layout: $ROOT/nfly (git clone) and $ROOT/fly-eeg (this folder), side by side
ROOT=${ROOT:-$HOME/flybrain}
cd "$ROOT/nfly"
export UV_CACHE_DIR=$ROOT/.uv-cache

ARTIFACT=${ARTIFACT:-eog}
PROTOCOL=${PROTOCOL:-centered}          # centered = SPAR-EEG protocol; whole = EEGdenoiseNet protocol
STATES=$ROOT/states_${ARTIFACT}_${PROTOCOL}

if [ "$PROTOCOL" = centered ]; then
    uv run --no-sync python ../fly-eeg/fly_eeg_denoise.py --protocol centered --artifact $ARTIFACT \
        --n-train 2400 --n-test 150 --batch 100 --no-shuffle-control --save-states "$STATES"
else
    uv run --no-sync python ../fly-eeg/fly_eeg_denoise.py --protocol whole --artifact $ARTIFACT \
        --n-train 4000 --n-test 500 --batch 100 --save-states "$STATES"
fi
uv run --no-sync python ../fly-eeg/fly_eeg_readout.py "$STATES" --steps 6000
