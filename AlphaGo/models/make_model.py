"""Create a model JSON - the architecture and input features a training run starts from.

    python -m AlphaGo.models.make_model newres --blocks 15 --filters 192
    python -m AlphaGo.models.make_model restower --blocks 15 --filters 192 --head conv_norm
    python -m AlphaGo.models.make_model cnn --layers 12 --filters 192

writes workspace/models/model_newres_b15c192_g5.json (model_restower_b15c192_convnorm.json,
model_cnn_l12c192.json),
ready for supervised_policy_trainer.py and lr_range_test.py. The JSON fixes the feature
planes the network reads, which must match the shards it trains on: by default the full
set convert_shuffled.py builds, or --features-from to copy them from existing shards.
"""
import argparse
import os
import sys

from AlphaGo.models.policy import CNNPolicy, NewResPolicy, ResTowerPolicy
from AlphaGo.preprocessing.convert_shuffled import ALL_FEATURES
from AlphaGo.training.shard_stream import dataset_info, find_split_shards

MODELS_DIRECTORY = os.path.join("workspace", "models")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    common = argparse.ArgumentParser(add_help=False)
    features = common.add_mutually_exclusive_group()
    features.add_argument("--features", help="Comma-separated input feature list. Default: "
                          "every feature convert_shuffled.py builds ({})".format(
                              ",".join(ALL_FEATURES)))
    features.add_argument("--features-from", metavar="SHARDS_DIR",
                          help="Use the feature list (and board size) of existing shards - "
                               "a convert_shuffled.py output directory - so the model is "
                               "guaranteed to match that data.")
    common.add_argument("--board", type=int, default=None,
                        help="Board size. Default: the shards' with --features-from, else 19")
    common.add_argument("--out", help="Output path. Default: {}/model_<architecture>.json, "
                        "named from the options".format(MODELS_DIRECTORY))
    common.add_argument("--force", action="store_true",
                        help="Overwrite an existing file. Refused by default: trained "
                             "weights only fit the exact model JSON they were trained with.")

    architectures = parser.add_subparsers(dest="architecture", required=True)

    newres = architectures.add_parser(
        "newres", parents=[common],
        help="NewResPolicy: a pre-activation residual tower with global pooling",
        description="NewResPolicy - see its create_network docstring for the options.")
    newres.add_argument("--blocks", type=int, required=True,
                        help="Residual blocks, each two conv layers")
    newres.add_argument("--filters", type=int, required=True,
                        help="Width of the residual stream")
    newres.add_argument("--gpool-every", type=int, default=5,
                        help="Every this-many-th block is a global pooling block; 0 for "
                             "none. Default: 5")
    newres.add_argument("--gpool-channels", type=int, default=64,
                        help="Channels a pooling block pools. Default: 64")
    newres.add_argument("--head-channels", type=int, default=32,
                        help="Width of the policy head's intermediate layer. Default: 32")
    newres.add_argument("--no-head-gpool", dest="head_gpool", action="store_false",
                        help="Leave out the policy head's pooled whole-board bias")
    newres.add_argument("--stem-filter-width", type=int, default=3,
                        help="Kernel size of the stem conv. Default: 3")

    restower = architectures.add_parser(
        "restower", parents=[common],
        help="ResTowerPolicy: an AlphaGo Zero-style residual tower",
        description="ResTowerPolicy - see its create_network docstring for the options.")
    restower.add_argument("--blocks", type=int, required=True,
                          help="Residual blocks, each two conv layers")
    restower.add_argument("--filters", type=int, required=True,
                          help="Width of the stem and every block")
    restower.add_argument("--head", choices=["conv", "conv_norm", "dense"], required=True,
                          help="Policy head: 'conv' (1x1 conv to 1 channel), 'conv_norm' "
                               "(widened and batch-normalized - the stable choice for deep "
                               "towers), 'dense' (AlphaGo Zero's)")
    restower.add_argument("--head-channels", type=int, default=32,
                          help="--head conv_norm only: width of its intermediate layer. "
                               "Default: 32")
    restower.add_argument("--stem-filter-width", type=int, default=3,
                          help="Kernel size of the stem conv. Default: 3")
    restower.add_argument("--block-filter-width", type=int, default=3,
                          help="Kernel size inside every block. Default: 3")

    cnn = architectures.add_parser(
        "cnn", parents=[common], help="CNNPolicy: a plain stack of convolutions",
        description="CNNPolicy - see its create_network docstring for the options.")
    cnn.add_argument("--layers", type=int, required=True, help="Convolutional layers")
    cnn.add_argument("--filters", type=int, required=True, help="Filters in every layer")
    cnn.add_argument("--first-filter-width", type=int, default=5,
                     help="Kernel size of the first layer (the rest are 3). Default: 5")
    cnn.add_argument("--kernel-initializer", default="uniform",
                     help="Conv kernel initializer, e.g. 'he_normal'. Default: 'uniform'")
    return parser


def _features_and_board(args):
    """The feature list and board size: from shards (--features-from), an explicit
    --features list, or the defaults."""
    if args.features_from:
        features, board, _n_planes, _sizes = dataset_info(
            find_split_shards(args.features_from, "train"))
        if args.board is not None and args.board != board:
            raise ValueError("--board {} doesn't match the shards' board size {}".format(
                args.board, board))
        return list(features), board
    features = args.features.split(",") if args.features else list(ALL_FEATURES)
    return features, args.board or 19


def _network(args):
    """(class, create_network kwargs, default file name) for the chosen architecture."""
    if args.architecture == "newres":
        kwargs = {"num_blocks": args.blocks, "filters": args.filters,
                  "gpool_every": args.gpool_every, "gpool_channels": args.gpool_channels,
                  "head_channels": args.head_channels, "head_gpool": args.head_gpool,
                  "stem_filter_width": args.stem_filter_width}
        name = "model_newres_b{}c{}_g{}{}.json".format(
            args.blocks, args.filters, args.gpool_every, "" if args.head_gpool else "_nohg")
        return NewResPolicy, kwargs, name
    if args.architecture == "restower":
        kwargs = {"num_blocks": args.blocks, "filters": args.filters, "head": args.head,
                  "head_channels": args.head_channels,
                  "stem_filter_width": args.stem_filter_width,
                  "block_filter_width": args.block_filter_width}
        name = "model_restower_b{}c{}_{}.json".format(
            args.blocks, args.filters, args.head.replace("_", ""))
        return ResTowerPolicy, kwargs, name
    kwargs = {"layers": args.layers, "filters_per_layer": args.filters,
              "filter_width_1": args.first_filter_width,
              "kernel_initializer": args.kernel_initializer}
    return CNNPolicy, kwargs, "model_cnn_l{}c{}.json".format(args.layers, args.filters)


def make_model(cmd_line_args=None):
    """Builds the model the arguments describe, writes its JSON, and returns the path."""
    args = build_parser().parse_args(cmd_line_args)
    features, board = _features_and_board(args)
    policy_class, kwargs, default_name = _network(args)
    out = args.out or os.path.join(MODELS_DIRECTORY, default_name)
    if os.path.exists(out) and not args.force:
        raise ValueError("{} already exists - trained weights only fit the exact model JSON "
                         "they were trained with. Choose another --out, or pass --force."
                         .format(out))

    policy = policy_class(features, board=board, **kwargs)
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    policy.save_model(out)

    print("{}: {} ({})".format(out, policy_class.__name__, ", ".join(
        "{}={}".format(k, v) for k, v in kwargs.items())))
    print("  board {}x{}, {} input planes from {} features: {}".format(
        board, board, policy.preprocessor.get_output_dimension(), len(features),
        ",".join(features)))
    print("  {:,} parameters".format(policy.model.count_params()))
    return out


if __name__ == "__main__":
    try:
        make_model()
    except ValueError as e:
        sys.exit("error: {}".format(e))
