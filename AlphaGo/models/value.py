"""A policy network with a value head: how good the position is for the player to move.

Value head after KataGo's ValueHead (python/katago/train/model_pytorch.py in KataGo
v1.18.2): a 1x1 conv, BN + ReLU, global mean and max pooling, then a dense layer to two
outputs -

  value  win probability for the player to move (sigmoid), trained on KataGo's search
         win rate for the position (see AlphaGo/preprocessing/add_value_targets.py)
  score  expected score lead for the player to move, in points

Two ways to build one:

- from_policy(): a value head added onto a trained policy network. The game's komi (player
  to move's side) is a second input that only the value head sees, so the trunk and policy
  stay the policy network's own; from_policy() freezes them, and set_trainable_blocks()
  unfreezes the top of the trunk for fine-tuning. See VALUE_HEAD_PLAN.md.
- PolicyValueNet(features, **kwargs) / create_network(): a joint network trained from
  scratch, NewResPolicy's trunk and policy head (any of its options, e.g. gpool_blocks
  [7, 12, 17]) with komi added to the trunk after the stem conv - so the policy sees it
  too - and a third output head:

  ownership  per point, who owns it at the end of the game, -1..1 for the player to move
             (tanh), trained on KataGo's ownership of the game's final position (see
             AlphaGo/preprocessing/add_ownership_targets.py)

  See JOINT_TRAINING_PLAN.md.

Outputs, in order: policy, value, score[, ownership].
"""
import numpy as np
from keras.initializers import VarianceScaling
from keras.layers import (Input, Add, BatchNormalization, Conv2D, Activation, Dense,
                          Concatenate, Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D,
                          Rescaling, Reshape)
from keras.models import Model

from AlphaGo.models.nn_util import neuralnet
from AlphaGo.models.policy import (CNNPolicy, newres_params, newres_policy_head,
                                   newres_trunk)

KOMI_SCALE = 20.0   # komi and score enter / leave the network divided / multiplied by this
VALUE = "value"
SCORE = "score"
OWNERSHIP = "ownership"


def _small():
    """Small initial output weights (as KataGo's value head): a new head starts out at
    about 50%, 0 points and no ownership, rather than at scores of +-100 points whose
    early error swamps the layers below."""
    return VarianceScaling(scale=0.01, mode="fan_in", distribution="truncated_normal")


def _value_features(x, channels):
    """1x1 conv, BN + ReLU: the value head's spatial features."""
    v = Conv2D(channels, 1, padding="same", use_bias=False,
               kernel_initializer="he_normal", name="value_conv")(x)
    v = BatchNormalization(name="value_bn")(v)
    return Activation("relu", name="value_relu")(v)


def _pooled(v):
    return Concatenate(name="value_pool")([GlobalAveragePooling2D(name="value_mean")(v),
                                           GlobalMaxPooling2D(name="value_max")(v)])


def _value_outputs(h):
    """value and score outputs from the head's hidden features h. Pinned to float32 under
    mixed precision, like the policy softmax (and kept so by NeuralNetBase.load_model,
    which keeps non-ReLU activations' dtype)."""
    value = Activation("sigmoid", dtype="float32", name=VALUE)(
        Dense(1, kernel_initializer=_small(), name="value_logit")(h))
    score = Activation("linear", dtype="float32", name=SCORE)(
        Rescaling(KOMI_SCALE, name="score_scale")(
            Dense(1, kernel_initializer=_small(), name="score_scaled")(h)))
    return value, score


def trunk_output(model):
    """The trunk's output tensor in a policy network: the input of the policy head's first
    1x1 conv. Every trunk conv (stem, blocks, pooling blocks) is 3x3, so the first 1x1 conv
    in the graph is where the head starts."""
    for layer in model.layers:
        if isinstance(layer, Conv2D) and tuple(layer.kernel_size) == (1, 1):
            return layer.input
    raise ValueError("no 1x1 conv in this model - not a policy network with a conv head")


def residual_adds(model):
    """The Add layers that end each residual block, in order: those taking the residual
    stream (the stem conv's output, then each previous block's sum) as an input - not a
    pooling block's internal add or the policy head's."""
    convs = [layer for layer in model.layers if isinstance(layer, Conv2D)]
    stream = convs[0].output.name
    adds = []
    for layer in model.layers:
        if isinstance(layer, Add) and stream in [t.name for t in layer.input]:
            adds.append(layer)
            stream = layer.output.name
    return adds


def is_value_layer(layer):
    return layer.name.startswith(("value", "score", "ownership"))


def set_trainable_blocks(net, n_blocks):
    """Train the value head plus the last n_blocks residual blocks and everything after
    them (the trunk's final norm and the policy head); freeze the rest. 0: value head
    only. n_blocks >= the number of blocks: the whole network."""
    adds = residual_adds(net.model)
    if n_blocks <= 0:
        first_trainable = len(net.model.layers)
    elif n_blocks >= len(adds):
        first_trainable = 0
    else:
        first_trainable = net.model.layers.index(adds[len(adds) - n_blocks - 1]) + 1
    for i, layer in enumerate(net.model.layers):
        layer.trainable = is_value_layer(layer) or i >= first_trainable


@neuralnet
class PolicyValueNet(CNNPolicy):
    """Policy + value + score [+ ownership]. forward() and eval_state() behave as a policy
    network's, so it drops in wherever one is used; forward_all() and eval_value() add the
    value outputs."""

    VALUE_DEFAULTS = {"value_channels": 48, "value_hidden": 112}  # KataGo's b20c256

    @staticmethod
    def create_network(**kwargs):
        """A joint network from scratch: NewResPolicy's keyword arguments (filters,
        num_blocks, gpool_blocks, ...) plus value_channels and value_hidden (default 48 and
        112, KataGo's b20c256 value head)."""
        head = dict(PolicyValueNet.VALUE_DEFAULTS)
        head.update({k: v for k, v in kwargs.items() if k in head})
        params = newres_params({k: v for k, v in kwargs.items() if k not in head})
        board = params["board"]
        planes = Input(shape=(board, board, params["input_dim"]))
        komi = Input(shape=(1,), name="komi")
        # komi -> a per-channel bias on the stem's output, at every point (KataGo's way of
        # feeding its global inputs into the trunk)
        k = Rescaling(1.0 / KOMI_SCALE, name="komi_scale")(komi)
        k = Dense(params["filters"], use_bias=False, kernel_initializer="he_normal",
                  name="komi_dense")(k)
        k = Reshape((1, 1, params["filters"]), name="komi_bias")(k)
        x = newres_trunk(params, planes, global_bias=k)

        policy = newres_policy_head(params, x)
        v = _value_features(x, head["value_channels"])
        h = Dense(head["value_hidden"], activation="relu", kernel_initializer="he_normal",
                  name="value_hidden")(_pooled(v))
        value, score = _value_outputs(h)
        own = Conv2D(1, 1, padding="same", use_bias=False, kernel_initializer=_small(),
                     name="ownership_conv")(v)
        own = Activation("tanh", dtype="float32", name=OWNERSHIP)(
            Flatten(name="ownership_flat")(own))
        return Model(inputs=[planes, komi], outputs=[policy, value, score, own])

    @classmethod
    def from_policy(cls, policy_net, value_channels=32, value_hidden=64, freeze=True):
        base = policy_net.model
        net = cls(policy_net.preprocessor.get_feature_list(), init_network=False,
                  board=int(base.inputs[0].shape[1]))
        if freeze:
            for layer in base.layers:
                layer.trainable = False
        planes = base.inputs[0]
        komi = Input(shape=(1,), name="komi")

        v = _value_features(trunk_output(base), value_channels)
        k = Rescaling(1.0 / KOMI_SCALE, name="value_komi_scale")(komi)
        h = Concatenate(name="value_features")([_pooled(v), k])
        h = Dense(value_hidden, activation="relu", kernel_initializer="he_normal",
                  name="value_hidden")(h)
        value, score = _value_outputs(h)

        net.model = Model(inputs=[planes, komi], outputs=[base.outputs[0], value, score])
        net.forward = net._model_forward()
        return net

    def _model_forward(self):
        """Planes (and optionally komi, player to move's side) -> policy, as for a policy
        network. Komi defaults to 0: a fine-tuned head's policy doesn't depend on it, but a
        joint network's does - pass it there."""
        def forward(planes, komi=None):
            if komi is None:
                komi = np.zeros(len(planes), np.float32)
            komi = np.asarray(komi, dtype=np.float32).reshape(-1, 1)
            return self.model([planes, komi], training=False)[0].numpy()
        return forward

    def forward_all(self, planes, komi):
        """(policy, value, score) arrays for a batch of planes and komi (player to move's
        side, + for White). The ownership output, if any, is left out."""
        komi = np.asarray(komi, dtype=np.float32).reshape(-1, 1)
        outputs = self.model([planes, komi], training=False)
        policy, value, score = outputs[0], outputs[1], outputs[2]
        return policy.numpy(), value.numpy()[:, 0], score.numpy()[:, 0]

    def eval_value(self, states, komis):
        """(value, score) for each GameState, from its player to move's side. komis: the
        game's komi from that same side (+komi when White is to move)."""
        planes = np.concatenate([self.preprocessor.state_to_tensor(s) for s in states])
        _policy, value, score = self.forward_all(planes, komis)
        return value, score
