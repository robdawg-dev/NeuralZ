# Built from uv.lock, so the container runs exactly the package versions local development
# and the tests use - Python included (.python-version). TensorFlow's CUDA/cuDNN libraries
# come from the lockfile too, as pip packages (the `gpu` extra: tensorflow[and-cuda]), so the
# host needs only the NVIDIA driver and the container toolkit.
#
# (The previous image was built on tensorflow/tensorflow:2.21.0-gpu. That fixed Python at
# 3.11, took TF/Keras/NumPy from the base image rather than from uv.lock, and shipped a
# mismatched system cuDNN 8.9 that had to be overridden with a manually registered
# nvidia-cudnn-cu12.)
FROM ubuntu:24.04

# build-essential: C++ compiler for the Cython-accelerated engine, the default
# AlphaGo.go / AlphaGo.preprocessing.preprocessing (see setup_cython.py).
RUN apt-get update \
 && apt-get install --no-install-recommends -y build-essential ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Same uv version as used locally.
COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /uvx /bin/

# The environment lives outside /workspace, which is where the repo is bind-mounted at
# run time and would otherwise hide it.
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH=/opt/venv/bin:$PATH

WORKDIR /opt/build
COPY pyproject.toml uv.lock .python-version ./
RUN uv python install \
 && uv sync --locked --no-install-project --extra gpu --extra gtp --extra viz --group dev \
 && rm -rf /root/.cache/uv

# Register every pip-installed CUDA library directory (site-packages/nvidia/*/lib) with the
# dynamic linker. TF 2.21's own RUNPATH lists only some of them (cublas, cudnn, cufft,
# cusparse, ...) and omits cusolver, nvjitlink, curand and nvrtc - without this,
# tf.config.list_physical_devices('GPU') comes back empty with "Could not load dynamic
# library 'libcusolver.so.11'" / "Cannot dlopen some GPU libraries". The old image never hit
# this because it had a system-wide CUDA toolkit.
RUN python -c "import glob, os, sysconfig; \
print('\n'.join(sorted(glob.glob(os.path.join(sysconfig.get_paths()['purelib'], 'nvidia', '*', 'lib')))))" \
    > /etc/ld.so.conf.d/nvidia-pip.conf \
 && ldconfig

WORKDIR /workspace
