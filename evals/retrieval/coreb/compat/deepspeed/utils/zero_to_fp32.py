"""Small DeepSpeed import shim for C2LLM inference.

The C2LLM model file imports this helper at module import time, but dense
retrieval only uses the model's encoder path and does not need ZeRO checkpoint
conversion. Keep the failure explicit if that conversion path is ever reached.
"""


def get_fp32_state_dict_from_zero_checkpoint(*args, **kwargs):
    raise RuntimeError(
        "DeepSpeed ZeRO checkpoint conversion is not available in this "
        "lightweight inference environment."
    )
