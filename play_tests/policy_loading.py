"""Shared model-loading for the play_tests/ scripts.

Keeps the mapping from a short model name (b20c256/b15c192/b10c128/2017net) to its
json/weights files in one place, so time_get_move.py, self_play_to_sgf.py, etc.
can't drift out of sync with each other about where these models actually live.
"""
import os

from AlphaGo.models.nn_util import NeuralNetBase

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

MODEL_SPECS = {
    # The name is also the model's directory under play_tests/models/.
    "b10c128": {
        # ResTowerPolicy, 10 blocks x 128 filters, conv_norm policy head. Minibatch 1024,
        # peak LR 0.1, plateau LR schedule; 125 epochs (~350M positions, ~23.3h), with a
        # manual LR halving at epoch 101 and a plateau-triggered one at epoch 116. From
        # benchmarks/_restower_b10c128_convnorm_mb1024_revert_seed90001, seed=90001.
        # Final/best (epoch 125): val_loss=1.6691, val_accuracy=0.5182.
        "json": "b10c128_model.json",
        "weights": "weights.00125.weights.h5",
    },
    "b15c192": {
        # ResTowerPolicy, 15 blocks x 192 filters, conv_norm policy head, on the 40M-position
        # data: minibatch 1024, LR 1.6, plateau schedule; 95 epochs. From
        # _restower_b15c192_v4shuf40m_mb1024_lr1p6_seed90001. This checkpoint (epoch 94):
        # val_loss=1.484, val_accuracy=0.556, val_top5_accuracy=0.874.
        "json": "b15c192_model.json",
        "weights": "weights.00094.weights.h5",
    },
    "b20c256": {
        # NewResPolicy (pre-activation residual tower with global pooling), 20 blocks x
        # 256 filters, policy head width 48 - trained on the 60M-position data
        # (workspace/prod_60m/shards) at minibatch 512, LR 1.6 with a 2,000-step warmup,
        # then 0.8/0.4/0.2/0.1/0.05 - see LR_STUDY_PLAN.md. From
        # workspace/runs/newres_b20c256. This checkpoint (epoch 67, final):
        # val_loss=1.3906, val_accuracy=0.5732, val_top5_accuracy=0.8921.
        "json": "b20c256_model.json",
        "weights": "weights.00067.weights.h5",
    },
    "2017net": {
        # The 2017 run: 12 layers x 192 filters, trained on KGS games. Converted from the
        # original legacy Keras 2.0.4 format by convert_2016net.py - the untouched
        # originals live in Data/Mamifreak/.
        "json": "model.json",
        "weights": "model.weights.h5",
    },
}


def load_policy(name):
    spec = MODEL_SPECS[name]
    model_dir = os.path.join(MODELS_DIR, name)
    json_path = os.path.join(model_dir, spec["json"])
    weights_path = os.path.join(model_dir, spec["weights"])
    policy = NeuralNetBase.load_model(json_path)
    policy.model.load_weights(weights_path)
    return policy
