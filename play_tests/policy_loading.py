"""Shared model-loading for the play_tests/ scripts.

Keeps the mapping from a short model name (simplecnn/resnet/2016net) to its
json/weights files in one place, so time_get_move.py, self_play_to_sgf.py, etc.
can't drift out of sync with each other about where these models actually live.
"""
import os

from AlphaGo.models.nn_util import NeuralNetBase
import AlphaGo.models.resnet_tower_policy  # noqa: F401 (registers CNNPolicy + ResTowerPolicy)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

MODEL_SPECS = {
    "simplecnn": {
        "json": "model.json",
        "weights": "weights.00007.weights.h5",
        "legacy": False,
    },
    "resnet": {
        "json": "model_restower.json",
        "weights": "weights.00007.weights.h5",
        "legacy": False,
    },
    "resnet2h": {
        # From the ~2hr/28-epoch bridge trial (benchmarks/_bridge_trial_seed5001,
        # seed=5001) run after fixing the validation-generator epoch-boundary bug -
        # same architecture as "resnet" but ~4x the training (28 epochs vs 7).
        "json": "model_restower.json",
        "weights": "weights.00028.weights.h5",
        "legacy": False,
    },
    "resnet2hkgs": {
        # Same architecture and recipe as "resnet2h" (28 epochs, same hyperparameters),
        # but trained on kgs-ugo-highdan (human KGS games) instead of katago-selfplay -
        # from benchmarks/_kgs_resnet_seed6001, seed=6001.
        "json": "model_restower.json",
        "weights": "weights.00028.weights.h5",
        "legacy": False,
    },
    "simplecnn2h": {
        # Same architecture as "simplecnn" but scaled from 7 to 28 epochs (~2hrs,
        # matching resnet2h's scale-up) on the same katago-selfplay data - from
        # benchmarks/_simplecnn_2h_seed7001, seed=7001.
        "json": "model.json",
        "weights": "weights.00028.weights.h5",
        "legacy": False,
    },
    "b10c128": {
        # ResTowerPolicy, 10 blocks x 128 filters, with the normalized/widened
        # policy head (head='conv_norm') added after the original 1-channel head was
        # found to collapse to random-baseline output under a large gradient event
        # mid-training - see genModel_restower_b10c128_convnorm.py. 207 epochs total
        # (~130.4M positions) on katago-selfplay: the original 28-epoch (~2hr) run,
        # then extended via --lr-schedule plateau (ReduceLROnPlateau, no fixed
        # horizon) for ~16 more hours across two resumes (fresh --seed each time,
        # since this codebase doesn't do true data-stream continuation) - from
        # benchmarks/_restower_b10c128_convnorm_plateau_seed8002. Final val_loss=1.8149
        # (best epoch 200: 1.8147), 11 LR cuts from peak 0.01 down to ~4.9e-6.
        "json": "model_restower_b10c128_convnorm.json",
        "weights": "weights.00207.weights.h5",
        "legacy": False,
    },
    "b10c128new": {
        # ResTowerPolicy, 10 blocks x 128 filters, same conv_norm head as "b10c128" -
        # rebuilt from scratch (not resumed) on the same katago-selfplay data with the
        # fixed mixed-precision load_model() bug (see nn_util.py), minibatch 512 instead
        # of 256, and peak LR 0.1 (from a corrected LR range test) instead of 0.01.
        # Stopped by request at epoch 46 (~128.8M positions, ~8.9h wall time) - the old
        # "b10c128" ran 207 epochs (~130.4M positions, ~14.2h); at matched cumulative
        # positions this run's val_accuracy already exceeds the old run's final result.
        # From benchmarks/_restower_b10c128_convnorm_mb512_lr01_seed11001, seed=11001.
        # Final val_loss=1.8470, val_accuracy=0.4959, LR=0.025 (mid-schedule, not annealed).
        "json": "model_restower_b10c128_convnorm.json",
        "weights": "weights.00046.weights.h5",
        "legacy": False,
    },
    "b10c128mb1024": {
        # ResTowerPolicy, 10 blocks x 128 filters, same conv_norm head as "b10c128"/
        # "b10c128new" - fully reverted trainer (no weight-decay/clipnorm/prefetch
        # changes, all from this session's investigation, kept), minibatch 1024
        # instead of 512, peak LR 0.1, plateau LR schedule. Ran 125 epochs (~350M
        # positions, ~23.3h wall time) with two LR halvings along the way: a manual
        # one (0.05->0.025 at epoch 101, forced via a metadata.json edit + --weights
        # resume) and a normal plateau-triggered one (0.025->0.0125 at epoch 116).
        # No NaN/divergence at any point - directly answers this session's open
        # question of whether the weight-norm divergence risk found in smaller-scale
        # diagnostics manifests at real production scale (it did not). From
        # benchmarks/_restower_b10c128_convnorm_mb1024_revert_seed90001, seed=90001.
        # Final/best (epoch 125): val_loss=1.6691, val_accuracy=0.5182.
        "json": "model_restower_b10c128_convnorm.json",
        "weights": "weights.00125.weights.h5",
        "legacy": False,
    },
    "b15c192": {
        # ResTowerPolicy, 15 blocks x 192 filters (this project's original/default
        # architecture size - see resnet_tower_policy.py's own class defaults), with
        # the same conv_norm head as the b10c128* family. Trained via
        # supervised_policy_trainer_v3.py with matched seed (90001) and matched
        # game_split.json (copied from b10c128mb1024's own run directory) against
        # that run, for a rigorous apples-to-apples comparison at equal positions
        # seen. Minibatch 1024, --lr-schedule plateau, peak LR 1.6. Two manual LR
        # halvings so far, each via a metadata.json edit + --weights resume with
        # --resume-warmup-steps (a 1-epoch linear ramp into the new LR - --weights
        # only reloads model weights, never optimizer state, so SGD momentum resets
        # to 0 on every resume; an un-cushioned LR jump onto that momentum-less
        # optimizer diverged to NaN the first time this was tried without a ramp):
        # 1.6->0.8 at epoch 35, and 0.8->0.4 at epoch 74. Seed also changed on each
        # resume (90001->90002->90003) since a resumed process's shuffle buffer
        # re-shuffles from a fresh pass-1 with no memory of how far the prior
        # process got through its own shuffle - reusing a seed already used by an
        # earlier launch would replay that same position order again. From
        # benchmarks/_restower_b15c192_convnorm_mb1024_lr1p6_seed90001. This
        # checkpoint (epoch 74, right at the second cut): val_loss=1.5608,
        # val_accuracy=0.5382, val_top5_accuracy=0.8633.
        "json": "model_restower_b15c192_convnorm.json",
        "weights": "weights.00074.weights.h5",
        "legacy": False,
    },
    "b15c192latest": {
            # Latest and greatest
            "json": "model_restower_b15c192_convnorm.json",
            "weights": "weights.00094.weights.h5",
            "legacy": False,
    },
    "2016net": {
        # Converted from the original legacy Keras 2.0.4 format by
        # convert_2016net.py - the untouched originals live in Data/Mamifreak/.
        "json": "model.json",
        "weights": "model.weights.h5",
        "legacy": False,
    },
}


def load_policy(name):
    spec = MODEL_SPECS[name]
    model_dir = os.path.join(MODELS_DIR, name)
    json_path = os.path.join(model_dir, spec["json"])
    weights_path = os.path.join(model_dir, spec["weights"])
    if spec["legacy"]:
        return NeuralNetBase.load_legacy_keras2_model(json_path, weights_file=weights_path)
    policy = NeuralNetBase.load_model(json_path)
    policy.model.load_weights(weights_path)
    return policy
