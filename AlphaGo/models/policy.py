from keras.models import Model, Sequential
from keras.layers import (Input, BatchNormalization, Conv2D, add, Activation, Flatten, Dense,
                          Concatenate, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape)
from keras.initializers import VarianceScaling
from AlphaGo.util import flatten_idx
from AlphaGo.models.nn_util import Bias, NeuralNetBase, neuralnet
import numpy as np


@neuralnet
class CNNPolicy(NeuralNetBase):
    """uses a convolutional neural network to evaluate the state of the game
    and compute a probability distribution over the next action
    """

    def _select_moves_and_normalize(self, nn_output, moves, size):
        """helper function to normalize a distribution over the given list of moves
        and return a list of (move, prob) tuples
        """
        if len(moves) == 0:
            return []
        move_indices = [flatten_idx(m, size) for m in moves]
        # get network activations at legal move locations
        distribution = nn_output[move_indices]
        distribution = distribution / distribution.sum()
        # list(), not a bare zip: callers (e.g. AlphaGo/ai.py) call len() on this and
        # may iterate it more than once, neither of which a Python 3 zip iterator supports.
        return list(zip(moves, distribution))

    def batch_eval_state(self, states, moves_lists=None):
        """Given a list of states, evaluates them all at once to make best use of GPU
        batching capabilities.

        Analogous to [eval_state(s) for s in states]

        Returns: a parallel list of move distributions as in eval_state
        """
        n_states = len(states)
        if n_states == 0:
            return []
        state_size = states[0].get_size()
        if not all([st.get_size() == state_size for st in states]):
            raise ValueError("all states must have the same size")
        # concatenate together all one-hot encoded states along the 'batch' dimension
        nn_input = np.concatenate([self.preprocessor.state_to_tensor(s) for s in states], axis=0)
        # pass all input through the network at once (backend makes use of
        # batches if len(states) is large)
        network_output = self.forward(nn_input)
        # default move lists to all legal moves
        moves_lists = moves_lists or [st.get_legal_moves() for st in states]
        results = [None] * n_states
        for i in range(n_states):
            results[i] = self._select_moves_and_normalize(network_output[i], moves_lists[i],
                                                          state_size)
        return results

    def eval_state(self, state, moves=None):
        """Given a GameState object, returns a list of (action, probability) pairs
        according to the network outputs

        If a list of moves is specified, only those moves are kept in the distribution
        """
        tensor = self.preprocessor.state_to_tensor(state)
        # run the tensor through the network
        network_output = self.forward(tensor)
        moves = moves or state.get_legal_moves()
        return self._select_moves_and_normalize(network_output[0], moves, state.get_size())

    @staticmethod
    def create_network(**kwargs):
        """construct a convolutional neural network.

        Keword Arguments:
        - input_dim:             depth of features to be processed by first layer (no default)
        - board:                 width of the go board to be processed (default 19)
        - filters_per_layer:     number of filters used on every layer (default 128)
        - filters_per_layer_K:   (where K is between 1 and <layers>) number of filters
                                 used on layer K (default #filters_per_layer)
        - layers:                number of convolutional steps (default 12)
        - filter_width_K:        (where K is between 1 and <layers>) width of filter on
                                 layer K (default 3 except 1st layer which defaults to 5).
                                 Must be odd.
        - kernel_initializer:    Conv2D kernel initializer. Default 'uniform' (original
                                 behavior, unchanged) - a plain, unscaled initializer with
                                 no variance scaling for network depth, unlike the He/
                                 Kaiming initialization standard practice for ReLU
                                 networks. Investigated as a candidate fix for a dead-ReLU
                                 collapse seen training this architecture (no BatchNorm to
                                 otherwise recenter pre-activations) on kgs-ugo-highdan
                                 data - an unscaled init can start some units already
                                 close to the edge of dying before any training even
                                 happens, on top of whatever risk a large early gradient
                                 update adds (confirmed via benchmarks/_dead_relu_check.py).
                                 Try 'he_normal' or 'he_uniform'.
        """
        defaults = {
            "board": 19,
            "filters_per_layer": 128,
            "layers": 12,
            "filter_width_1": 5,
            "kernel_initializer": "uniform"
        }
        # copy defaults, but override with anything in kwargs
        params = defaults
        params.update(kwargs)

        # create the network:
        # a series of zero-paddings followed by convolutions
        # such that the output dimensions are also board x board
        network = Sequential()

        # create first layer
        network.add(Conv2D(
            input_shape=(params["board"], params["board"], params["input_dim"]),
            filters=params.get("filters_per_layer_1", params["filters_per_layer"]),
            kernel_size=(params["filter_width_1"], params["filter_width_1"]),
            kernel_initializer=params["kernel_initializer"],
            activation='relu',
            padding='same',
            kernel_constraint=None,
            activity_regularizer=None,
            trainable=True,
            strides=[1, 1],
            use_bias=True,
            bias_regularizer=None,
            bias_constraint=None,
            data_format="channels_last",
            kernel_regularizer=None))

        # create all other layers
        for i in range(2, params["layers"] + 1):
            # use filter_width_K if it is there, otherwise use 3
            filter_key = "filter_width_%d" % i
            filter_width = params.get(filter_key, 3)

            # use filters_per_layer_K if it is there, otherwise use default value
            filter_count_key = "filters_per_layer_%d" % i
            filter_nb = params.get(filter_count_key, params["filters_per_layer"])

            network.add(Conv2D(
                filters=filter_nb,
                kernel_size=(filter_width, filter_width),
                kernel_initializer=params["kernel_initializer"],
                activation='relu',
                padding='same',
                kernel_constraint=None,
                activity_regularizer=None,
                trainable=True,
                strides=[1, 1],
                use_bias=True,
                bias_regularizer=None,
                bias_constraint=None,
                data_format="channels_last",
                kernel_regularizer=None))

        # the last layer maps each <filters_per_layer> feature to a number
        network.add(Conv2D(
            filters=1,
            kernel_size=(1, 1),
            kernel_initializer='uniform',
            padding='same',
            kernel_constraint=None,
            activity_regularizer=None,
            trainable=True,
            strides=[1, 1],
            use_bias=True,
            bias_regularizer=None,
            bias_constraint=None,
            data_format="channels_last",
            kernel_regularizer=None))

        # reshape output to be board x board
        network.add(Flatten())
        # add a bias to each board location
        network.add(Bias())
        # softmax makes it into a probability distribution. Forced to float32 regardless
        # of a global mixed-precision policy - softmax over many classes plus the loss
        # computation right after it are the standard spot mixed precision recommends
        # keeping at full precision, to avoid overflow/underflow right at the output.
        network.add(Activation('softmax', dtype='float32'))

        return network


@neuralnet
class ResTowerPolicy(CNNPolicy):
    """Fixed-width residual tower policy network, a la AlphaGo Zero's architecture
    (Silver et al. 2017).

    Differs from CNNPolicy (a plain stack of same-width convs, no skip connections): each
    block is the AlphaGo Zero block - Conv-BN-ReLU-Conv-BN-(add identity)-ReLU, POST-
    activation, uniform width throughout - configured with two params (num_blocks,
    filters) instead of a kwarg per layer.

    Stays within this codebase's existing action space (board*board, no separate 'pass'
    class) and reuses the existing lightweight policy head (1x1 conv down to 1 channel +
    learned per-position bias + softmax) by default. AlphaGo Zero's own head (1x1 conv to
    2 channels + BN + ReLU + a dense layer to the full action space) is available via
    head='dense' for anyone who wants the extra head capacity, and a normalized/widened
    variant of the default lightweight head is available via head='conv_norm' (see its
    docstring in create_network for the stability rationale) - either way the action
    space is unchanged, so nothing downstream (ai.py, the trainer,
    shard_stream.encode's labels) needs to change.
    """

    @staticmethod
    def create_network(**kwargs):
        """
        Keyword Arguments:
        - input_dim:            depth of features processed by the stem (no default)
        - board:                 width of the go board (default 19)
        - filters:                width of the stem and every block in the tower
                                  (default 192 - use 256 for a closer match to AlphaGo
                                  Zero's own 20x256, compute permitting)
        - num_blocks:            number of residual blocks, each with 2 conv layers
                                  (default 15 - use 20 for AlphaGo Zero's own depth)
        - stem_filter_width:     kernel size of the stem conv (default 3, matching
                                  AlphaGo Zero; CNNPolicy defaults its first layer to
                                  5 instead)
        - block_filter_width:    kernel size used inside every block (default 3)
        - head:                  'conv' (default - this codebase's existing lightweight
                                  head, a single 1x1 conv straight down to 1 channel),
                                  'dense' (AlphaGo Zero's own head), or 'conv_norm' (a
                                  general resnet-head stability fix - see below).
        - head_channels:         width of the intermediate layer for head='conv_norm'
                                  (default 32)

        head='conv_norm' - motivation: diagnosed after a b10c128 katago-selfplay run
        collapsed to random-baseline output mid-training. The 20-block tower itself
        tested completely healthy (no dead units, sane BatchNorm statistics throughout),
        but its activation variance legitimately grows across the tower (measured up to
        40-60 by the deepest blocks - normal for a post-activation resnet with no
        re-normalization of the residual stream). head='conv' feeds that large,
        unnormalized signal straight into a single-channel linear projection (only
        `filters`+1 parameters total) - an extreme bottleneck with nowhere to absorb an
        unusually large or confidently-wrong batch, so one bad gradient can dominate and
        collapse the whole head toward the "safe" (but useless) uniform-output optimum.
        head='conv_norm' widens the projection to `head_channels` first, re-normalizes
        with BatchNorm+ReLU (matching how every other point in the tower controls its own
        scale), then makes the final 1x1 projection with deliberately smaller initial
        weights (variance-scaled down by 0.3x) so it starts calibrated rather than
        producing overly large initial logits. This mirrors general resnet/output-head
        stability practice (comparable structure appears in KataGo's own policy head) -
        it does NOT add anything KataGo-specific like global board-context pooling, a
        value head, or pass-move handling; the action space and everything downstream
        (ai.py, the trainer, shard_stream.encode's labels) is unchanged.
        """
        defaults = {
            "board": 19,
            "filters": 192,
            "num_blocks": 15,
            "stem_filter_width": 3,
            "block_filter_width": 3,
            "head": "conv",
            "head_channels": 32,
        }
        params = dict(defaults)
        params.update(kwargs)
        board = params["board"]
        filters = params["filters"]

        model_input = Input(shape=(board, board, params["input_dim"]))

        # Stem: plain conv + BN + ReLU - not itself a residual block, matching AlphaGo
        # Zero's published architecture. he_normal (not this codebase's usual 'uniform'):
        # standard practice for deep ReLU stacks - BatchNorm reduces but doesn't eliminate
        # the value of variance-scaled init at this depth (15-20 blocks x 2 convs each).
        x = Conv2D(filters, params["stem_filter_width"], padding="same",
                   kernel_initializer="he_normal", use_bias=True,
                   data_format="channels_last")(model_input)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # Tower: num_blocks residual blocks, each Conv-BN-ReLU-Conv-BN-(add)-ReLU.
        for _ in range(params["num_blocks"]):
            block_input = x
            x = Conv2D(filters, params["block_filter_width"], padding="same",
                       kernel_initializer="he_normal", use_bias=True,
                       data_format="channels_last")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Conv2D(filters, params["block_filter_width"], padding="same",
                       kernel_initializer="he_normal", use_bias=True,
                       data_format="channels_last")(x)
            x = BatchNormalization()(x)
            x = add([block_input, x])
            x = Activation("relu")(x)

        if params["head"] == "dense":
            x = Conv2D(2, 1, padding="same", kernel_initializer="uniform",
                       use_bias=True, data_format="channels_last")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Flatten()(x)
            x = Dense(board * board, kernel_initializer="uniform")(x)
        elif params["head"] == "conv_norm":
            x = Conv2D(params["head_channels"], 1, padding="same",
                       kernel_initializer="he_normal", use_bias=True,
                       data_format="channels_last")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            # scale=0.6 -> He-normal (scale=2.0) with 0.3x the usual variance, so this
            # final projection starts with deliberately small, well-calibrated weights
            # instead of the same init scale as every other layer - see the head='conv_norm'
            # docstring above for why.
            x = Conv2D(1, 1, padding="same",
                       kernel_initializer=VarianceScaling(
                           scale=0.6, mode="fan_in", distribution="truncated_normal"),
                       use_bias=True, data_format="channels_last")(x)
            x = Flatten()(x)
            x = Bias()(x)
        else:
            x = Conv2D(1, 1, padding="same", kernel_initializer="uniform",
                       use_bias=True, data_format="channels_last")(x)
            x = Flatten()(x)
            x = Bias()(x)

        # Forced to float32 regardless of a global mixed-precision policy - same reason
        # as policy.py's heads: keep the softmax/loss computation at full precision.
        output = Activation("softmax", dtype="float32")(x)

        return Model(inputs=[model_input], outputs=[output])


def _global_bias(x, out_channels):
    """KataGo-style global pooling: x's channels, normalized, pooled to their mean and max
    over the whole board, and mapped to a per-channel bias of width out_channels - to be
    added at every point of a spatial feature map. Gives every point
    whole-board context (ko, the status of large groups, who is ahead) in one step, where
    3x3 convolutions spread information one point per layer."""
    g = BatchNormalization()(x)
    g = Activation("relu")(g)
    pooled = Concatenate()([GlobalAveragePooling2D()(g), GlobalMaxPooling2D()(g)])
    bias = Dense(out_channels, use_bias=False, kernel_initializer="he_normal")(pooled)
    return Reshape((1, 1, out_channels))(bias)


@neuralnet
class NewResPolicy(CNNPolicy):
    """Pre-activation residual tower with global pooling, after KataGo's convnets
    (python/katago/train/model_pytorch.py in KataGo v1.18.2).

    Differs from ResTowerPolicy (AlphaGo Zero's post-activation tower):
    - Pre-activation blocks, x + conv(relu(BN(conv(relu(BN(x)))))): the residual stream is
      a clean identity path, and one BN + ReLU after the last block normalizes it before
      the head. ResTowerPolicy's stream grows unnormalized through the tower (the reason
      its head='conv_norm' exists).
    - Global pooling blocks: in every gpool_every-th block the first conv is split into
      filters - gpool_channels regular channels and gpool_channels pooled ones, whose
      board-wide mean and max become a per-channel bias on the regular ones (see
      _global_bias). KataGo's b15c192 has two such blocks; its recommended nets one in
      three. KataGo's third pooled value, a mean scaled by board size, is constant on a
      fixed board and left out.
    - The policy head, KataGo's shape: a 1x1 conv to head_channels, plus (head_gpool) a
      pooled whole-board bias, then BN + ReLU and a 1x1 conv to one channel with small
      initial weights. No pass output (this codebase's action space has none).
    - No conv biases - every conv output reaches a BN, which removes them.

    Kept as in ResTowerPolicy: BatchNorm in every block (KataGo's current nets instead use
    a single norm at the end of the trunk with fixed-scale initialization and adaptive
    weight decay), ReLU, he_normal initialization, the policy-only output over
    board*board points.
    """

    @staticmethod
    def create_network(**kwargs):
        """
        Keyword Arguments:
        - input_dim:          depth of features processed by the stem (no default)
        - board:              width of the go board (default 19)
        - filters:            width of the residual stream (default 192)
        - num_blocks:         number of residual blocks, each with 2 conv layers
                              (default 15)
        - gpool_every:        every this-many-th block is a global pooling block, counting
                              from the first (default 5: blocks 5, 10, 15 of 15); 0 for none
        - gpool_channels:     channels of a pooling block's first conv that are pooled
                              (default 64, KataGo's b15c192 value); the other
                              filters - gpool_channels stay spatial
        - head_channels:      width of the policy head's intermediate layer (default 32)
        - head_gpool:         whether the policy head adds a pooled whole-board bias
                              (default True)
        - stem_filter_width:  kernel size of the stem conv (default 3)
        """
        defaults = {
            "board": 19,
            "filters": 192,
            "num_blocks": 15,
            "gpool_every": 5,
            "gpool_channels": 64,
            "head_channels": 32,
            "head_gpool": True,
            "stem_filter_width": 3,
        }
        params = dict(defaults)
        params.update(kwargs)
        board = params["board"]
        filters = params["filters"]
        gpool_channels = params["gpool_channels"]
        if params["gpool_every"] and not 0 < gpool_channels < filters:
            raise ValueError("gpool_channels must be between 0 and filters ({}), got {}"
                             .format(filters, gpool_channels))

        def conv(channels, width, x):
            return Conv2D(channels, width, padding="same", use_bias=False,
                          kernel_initializer="he_normal", data_format="channels_last")(x)

        model_input = Input(shape=(board, board, params["input_dim"]))

        # Stem: a plain conv straight into the residual stream - the first block's BN + ReLU
        # normalizes it.
        x = conv(filters, params["stem_filter_width"], model_input)

        for i in range(1, params["num_blocks"] + 1):
            h = BatchNormalization()(x)
            h = Activation("relu")(h)
            if params["gpool_every"] and i % params["gpool_every"] == 0:
                regular = conv(filters - gpool_channels, 3, h)
                pooled = conv(gpool_channels, 3, h)
                h = add([regular, _global_bias(pooled, filters - gpool_channels)])
            else:
                h = conv(filters, 3, h)
            h = BatchNormalization()(h)
            h = Activation("relu")(h)
            h = conv(filters, 3, h)
            x = add([x, h])

        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        head = params["head_channels"]
        p = conv(head, 1, x)
        if params["head_gpool"]:
            p = add([p, _global_bias(conv(head, 1, x), head)])
        p = BatchNormalization()(p)
        p = Activation("relu")(p)
        # scale=0.6 -> He-normal (scale=2.0) with 0.3x the usual variance, so the output
        # starts with small logits - as ResTowerPolicy's head='conv_norm' and KataGo's head.
        p = Conv2D(1, 1, padding="same", use_bias=True, data_format="channels_last",
                   kernel_initializer=VarianceScaling(
                       scale=0.6, mode="fan_in", distribution="truncated_normal"))(p)
        p = Flatten()(p)
        p = Bias()(p)

        # Forced to float32 regardless of a global mixed-precision policy - same reason
        # as CNNPolicy's: keep the softmax/loss computation at full precision.
        output = Activation("softmax", dtype="float32")(p)

        return Model(inputs=[model_input], outputs=[output])
