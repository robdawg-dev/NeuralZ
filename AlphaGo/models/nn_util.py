from keras import backend as K
from keras.models import model_from_json
from keras.layers import Layer
from AlphaGo.preprocessing.preprocessing import Preprocess
import json


class NeuralNetBase(object):
    """Base class for neural network classes handling feature processing, construction
    of a 'forward' function, etc.
    """

    # keep track of subclasses to make generic saving/loading cleaner.
    # subclasses can be 'registered' with the @neuralnet decorator
    subclasses = {}

    def __init__(self, feature_list, **kwargs):
        """create a neural net object that preprocesses according to feature_list and uses
        a neural network specified by keyword arguments (using subclass' create_network())

        optional argument: init_network (boolean). If set to False, skips initializing
        self.model and self.forward and the calling function should set them.
        """
        # Preprocess's own board-size default (19) must match whatever 'board' size the
        # network is built for - otherwise Preprocess iterates/indexes assuming a 19x19
        # board while GameState uses a different size, which segfaults (out-of-bounds
        # access into the Cython engine's board vector) rather than raising a Python error.
        self.preprocessor = Preprocess(feature_list, size=kwargs.get('board', 19))
        kwargs["input_dim"] = self.preprocessor.get_output_dimension()

        if kwargs.get('init_network', True):
            # self.__class__ refers to the subclass so that subclasses only
            # need to override create_network()
            self.model = self.__class__.create_network(**kwargs)
            # self.forward is a lambda function wrapping a Keras function
            self.forward = self._model_forward()

    def _model_forward(self):
        """Construct a function using the current keras backend that, when given a batch
        of inputs, simply processes them forward and returns the output

        This is as opposed to model.compile(), which takes a loss function
        and training method.

        c.f. https://github.com/fchollet/keras/issues/1426
        """
        # Always run in inference mode (equivalent to the old learning-phase-aware
        # K.function branch, which no longer exists in this Keras version).
        #
        # A direct eager call (not model.predict()) is used deliberately: predict()'s
        # compiled/XLA path hits the same GPU autotuner failure as fit() (see the
        # jit_compile=False fix in the training scripts) for this architecture, even
        # for a model that was never explicitly .compile()'d. Calling the model
        # directly runs eagerly and sidesteps XLA's fused conv-bias-activation kernel
        # selection entirely.
        return lambda inpt: self.model(inpt, training=False).numpy()

    @staticmethod
    def load_model(json_file):
        """create a new neural net object from the architecture specified in json_file
        """
        with open(json_file, 'r') as f:
            object_specs = json.load(f)

        # Create object; may be a subclass of networks saved in specs['class']
        class_name = object_specs.get('class', 'CNNPolicy')
        try:
            network_class = NeuralNetBase.subclasses[class_name]
        except KeyError:
            raise ValueError("Unknown neural network type in json file: {}\n"
                             "(was it registered with the @neuralnet decorator?)"
                             .format(class_name))

        # create new object
        new_net = network_class(object_specs['feature_list'], init_network=False)

        # Every layer's dtype policy gets baked into its serialized config at save time
        # (whatever mixed_precision policy was active when the model was originally
        # built), and model_from_json() faithfully reconstructs that literal dtype -
        # it does NOT defer to whatever global policy is active now, at load time. Left
        # alone, this makes mixed_precision.set_global_policy("mixed_float16") a no-op
        # for any model loaded from a JSON that was saved under plain float32 (i.e.
        # every model.json this project has ever generated) - confirmed by comparing a
        # loaded model's per-layer compute_dtype against a freshly-constructed one under
        # the same global policy: every layer past the input stayed "float32" after
        # loading, while direct construction correctly picked up "mixed_float16".
        # Strip the baked-in dtype from every layer except the one deliberate exception
        # in this codebase's own architecture code (policy.py): the softmax activation
        # is intentionally pinned to float32 regardless of the global policy, for
        # numerical stability. Everything else should track the current policy, matching
        # how a freshly-built model behaves.
        keras_model_config = json.loads(object_specs['keras_model'])
        for layer_config in keras_model_config['config']['layers']:
            layer_cfg = layer_config.get('config', {})
            is_pinned_softmax = (layer_config.get('class_name') == 'Activation' and
                                 layer_cfg.get('activation') == 'softmax')
            if not is_pinned_softmax:
                layer_cfg.pop('dtype', None)

        new_net.model = model_from_json(json.dumps(keras_model_config),
                                        custom_objects={'Bias': Bias})
        if 'weights_file' in object_specs:
            new_net.model.load_weights(object_specs['weights_file'])
        new_net.forward = new_net._model_forward()
        return new_net

    @staticmethod
    def load_legacy_keras2_model(json_file, weights_file=None):
        """Load a model saved by an old (pre-Keras 3, e.g. Data/Mamifreak's Keras 2.0.4)
        version of save_model(), for which load_model() doesn't work: Keras 3's
        model_from_json refuses to deserialize the old Sequential JSON schema outright
        ("Could not locate class 'Sequential'").

        Since the embedded architecture can't be deserialized, this instead parses the
        embedded Keras 2 JSON as plain data (never through model_from_json) to recover
        just the Conv2D layer sizes, then rebuilds an equivalent network from scratch
        using this codebase's own current create_network() (channels_last, like every
        other model here - the original's data_format is irrelevant since Keras stores
        Conv2D kernels as (kh, kw, in_ch, out_ch) regardless of data_format, so no weight
        permutation is needed). Weights are then loaded onto that fresh model directly
        via load_weights(), which for this legacy per-layer HDF5 format matches by layer
        order rather than needing exact name equality - confirmed working (sensible,
        non-uniform move-probability output, not e.g. all-uniform/garbage) against
        Data/Mamifreak's 48-plane/192-filter/12-layer policy network.

        Only supports the plain CNNPolicy/ResnetPolicy-style stack this repo has always
        produced (N x Conv2D, then a final 1-filter/1x1 Conv2D, Flatten, Bias, softmax
        Activation) - raises ValueError if the file doesn't look like that, rather than
        silently building the wrong architecture.
        """
        with open(json_file, 'r') as f:
            object_specs = json.load(f)

        class_name = object_specs.get('class', 'CNNPolicy')
        try:
            network_class = NeuralNetBase.subclasses[class_name]
        except KeyError:
            raise ValueError("Unknown neural network type in json file: {}\n"
                             "(was it registered with the @neuralnet decorator?)"
                             .format(class_name))

        keras_config = json.loads(object_specs['keras_model'])
        layer_configs = keras_config['config']
        # Some Keras JSON variants nest layers under a 'layers' key instead of being the
        # config list itself.
        if isinstance(layer_configs, dict) and 'layers' in layer_configs:
            layer_configs = layer_configs['layers']

        conv_layers = [c for c in layer_configs if c['class_name'] == 'Conv2D']
        tail_classes = [c['class_name'] for c in layer_configs if c['class_name'] != 'Conv2D']
        if tail_classes != ['Flatten', 'Bias', 'Activation'] or len(conv_layers) < 2 or \
                conv_layers[-1]['config']['filters'] != 1:
            raise ValueError(
                "{} doesn't look like the plain Conv2D*N + Flatten + Bias + Activation "
                "stack this loader supports (found layer classes: {})"
                .format(json_file, [c['class_name'] for c in layer_configs]))

        # batch_input_shape is [batch, channels, H, W] for channels_first or
        # [batch, H, W, channels] for channels_last - use the layer's own data_format
        # rather than assuming, so this doesn't silently mis-read a channels_last legacy
        # file (this codebase's own history only ever produced channels_first ones, but
        # don't bake that assumption into an index).
        input_shape = conv_layers[0]['config']['batch_input_shape']
        data_format = conv_layers[0]['config'].get('data_format', 'channels_first')
        board = input_shape[2] if data_format == 'channels_first' else input_shape[1]
        params = {"board": board, "layers": len(conv_layers) - 1}
        # conv_layers[:-1] are the "regular" layers create_network() builds via its
        # filters_per_layer_K/filter_width_K kwargs; the always-present final 1x1/1-filter
        # output layer is added by create_network() itself and isn't parameterized here.
        for i, conv in enumerate(conv_layers[:-1], start=1):
            params["filters_per_layer_%d" % i] = conv['config']['filters']
            params["filter_width_%d" % i] = conv['config']['kernel_size'][0]

        new_net = network_class(object_specs['feature_list'], **params)

        weights_file = weights_file or object_specs.get('weights_file')
        if weights_file:
            new_net.model.load_weights(weights_file)
        return new_net

    def save_model(self, json_file, weights_file=None):
        """write the network model and preprocessing features to the specified file

        If a weights_file (.hdf5 extension) is also specified, model weights are also
        saved to that file and will be reloaded automatically in a call to load_model
        """
        # this looks odd because we are serializing a model with json as a string
        # then making that the value of an object which is then serialized as
        # json again.
        # It's not as crazy as it looks. A Network has 2 moving parts - the
        # feature preprocessing and the neural net, each of which gets a top-level
        # entry in the saved file. Keras just happens to serialize models with JSON
        # as well. Note how this format makes load_model fairly clean as well.
        object_specs = {
            'class': self.__class__.__name__,
            'keras_model': self.model.to_json(),
            'feature_list': self.preprocessor.get_feature_list()
        }
        if weights_file is not None:
            self.model.save_weights(weights_file)
            object_specs['weights_file'] = weights_file
        # use the json module to write object_specs to file
        with open(json_file, 'w') as f:
            json.dump(object_specs, f)


def neuralnet(cls):
    """Class decorator for registering subclasses of NeuralNetBase
    """
    NeuralNetBase.subclasses[cls.__name__] = cls
    return cls


class Bias(Layer):
    """Custom keras layer that simply adds a scalar bias to each location in the input

    Largely copied from the keras docs:
    http://keras.io/layers/writing-your-own-keras-layers/#writing-your-own-keras-layers
    """

    def __init__(self, **kwargs):
        super(Bias, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='bias', shape=input_shape[1:],
                                 initializer='zeros', trainable=True)

    def call(self, x, mask=None):
        return x + self.W
