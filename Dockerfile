# TensorFlow's official image bundles a matched CUDA toolkit for this exact TF version,
# so there's no manual CUDA install to get right on top of the host driver. It DOES ship
# a mismatched system cuDNN 8.9 though, while this TF build needs cuDNN 9 (dlopen's
# libcudnn.so.9) - confirmed by `tf.config.list_physical_devices('GPU')` returning empty
# with "Cannot dlopen some GPU libraries" until nvidia-cudnn-cu12 (which ships cuDNN 9) is
# installed and registered below.
FROM tensorflow/tensorflow:2.21.0-gpu

# build-essential: C++ compiler for the Cython-accelerated engine, the default
# AlphaGo.go / AlphaGo.preprocessing.preprocessing (see setup_cython.py).
RUN apt-get update \
 && apt-get install --no-install-recommends -y build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "h5py>=3.11.0,<3.15.0" \
    scipy \
    PyYAML \
    sgf==0.5 \
    pygtp==0.3 \
    Cython \
    pytest \
    nvidia-cudnn-cu12==9.5.1.17 \
 && CUDNN_LIB_DIR=$(python3 -c "import nvidia.cudnn, os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'lib'))") \
 && echo "$CUDNN_LIB_DIR" > /etc/ld.so.conf.d/nvidia-cudnn.conf \
 && ldconfig

WORKDIR /workspace
