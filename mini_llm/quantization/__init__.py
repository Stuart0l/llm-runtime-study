"""Quantized projection backends."""

from mini_llm.quantization.gptq_marlin import (
    GPTQMarlinLinear,
    validate_gptq_marlin_device,
)

__all__ = ["GPTQMarlinLinear", "validate_gptq_marlin_device"]
