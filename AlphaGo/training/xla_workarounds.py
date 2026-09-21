"""Defensive workarounds for XLA/cuDNN issues found on this project's training GPU.

Kept in one place and imported by anything using jit_compile=True, rather than relying
solely on docker-compose.yml's XLA_FLAGS env var - that env var is invisible to (and
silently lost by) anyone running training code outside that exact compose config, and the
failure mode if it's missing is a hard crash on the very first training step.
"""
import os
import sys
import warnings

_CONV_NHWC_FIX = "--xla_gpu_force_conv_nhwc=true"


def ensure_xla_conv_nhwc():
    """Force NHWC layout for cuDNN convolutions under XLA.

    Without this, XLA's autotuner fails to find any supported cuDNN algorithm for this
    model's fused conv+bias+activation op on this GPU/driver/cuDNN9 combination
    (RTX 4070 SUPER / Ada, CUDA 12.5.1, cuDNN 9) - it picks an NCHW layout (a transpose
    away from this model's native NHWC/channels_last) for the fused op, and no algorithm
    supports that transposed shape ("Autotuner could not find any supported configs for
    HLO: ...convBiasActivationForward"). Confirmed via a minimal repro.

    MUST RUN BEFORE THE FIRST XLA COMPILATION. In practice that means before anything
    triggers a jit-compiled op - model.compile(jit_compile=True) plus the fit/predict call
    that exercises it, or any other jit'd computation in the process.

    An earlier version of this docstring claimed the flag "still takes effect even after
    TF is already imported, since XLA reads XLA_FLAGS lazily at first compile". That is not
    reliable, and the failure is a hard crash rather than a silent fallback: a harness that
    imported TensorFlow and ran a Keras evaluate() before importing this module hit exactly
    the autotuner error above, with the HLO showing dim_labels=bf01_oi01->bf01 (NCHW) -
    i.e. the flag had not applied. Import order matters. Production entry points are fine
    because they import the trainer (and therefore this module) first; anything else should
    import this module before touching TensorFlow.

    Emits a warning if TensorFlow is already imported, since that is the situation where
    the flag may arrive too late.
    """
    already_set = _CONV_NHWC_FIX in os.environ.get("XLA_FLAGS", "")
    if not already_set:
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "") + " " + _CONV_NHWC_FIX).strip()
        if "tensorflow" in sys.modules:
            warnings.warn(
                "ensure_xla_conv_nhwc() ran after TensorFlow was already imported. "
                "XLA may have parsed its flags already, in which case "
                "{} will not apply and the first jit_compile=True step can fail with "
                "'Autotuner could not find any supported configs'. Import "
                "AlphaGo.training.xla_workarounds (or the trainer module) before "
                "TensorFlow, or set XLA_FLAGS in the environment.".format(_CONV_NHWC_FIX),
                RuntimeWarning, stacklevel=2)
