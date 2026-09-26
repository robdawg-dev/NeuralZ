from keras.models import Model
from keras.layers import Input, BatchNormalization, Conv2D, add, Activation, Flatten, Dense
from keras.initializers import VarianceScaling
from AlphaGo.models.nn_util import Bias, neuralnet
from AlphaGo.models.policy import CNNPolicy


@neuralnet
class ResTowerPolicy(CNNPolicy):
    """Fixed-width residual tower policy network, a la AlphaGo Zero's architecture
    (Silver et al. 2017) - a new class alongside CNNPolicy/ResnetPolicy (both left
    untouched), so existing trained models keep working unchanged.

    Differs from CNNPolicy (a plain stack of same-width convs, no skip connections) and
    from ResnetPolicy (skip connections, but each "unit" is a single conv wrapped in a
    residual connection, configured per-layer via n_skip_K kwargs - a real N-block tower
    needs n_skip_K=2 set by hand on N different kwargs). Here each block is hardcoded as
    the actual AlphaGo Zero block - Conv-BN-ReLU-Conv-BN-(add identity)-ReLU, POST-
    activation, uniform width throughout - configured with two params (num_blocks,
    filters) instead of a kwarg per layer. (ResnetPolicy's BN-ReLU-Conv ordering is
    PRE-activation, He et al. 2016's later variant - a legitimate, arguably better choice
    for very deep nets, but not what "a la AlphaGo Zero" specifically means.)

    Stays within this codebase's existing action space (board*board, no separate 'pass'
    class) and reuses the existing lightweight policy head (1x1 conv down to 1 channel +
    learned per-position bias + softmax) by default. AlphaGo Zero's own head (1x1 conv to
    2 channels + BN + ReLU + a dense layer to the full action space) is available via
    head='dense' for anyone who wants the extra head capacity, and a normalized/widened
    variant of the default lightweight head is available via head='conv_norm' (see its
    docstring in create_network for the stability rationale) - either way the action
    space is unchanged, so nothing downstream (ai.py, mcts.py, the trainer,
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
                                  AlphaGo Zero; CNNPolicy/ResnetPolicy default their
                                  first layer to 5 instead)
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
        (ai.py, mcts.py, the trainer, shard_stream.encode's labels) is unchanged.
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
