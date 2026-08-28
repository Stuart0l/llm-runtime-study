"""Architecture registry for loading supported causal language models."""

from __future__ import annotations

from pathlib import Path

from mini_llm.config import ConfigError, DecoderConfig, load_config
from mini_llm.granite_model import GraniteMoeForCausalLM
from mini_llm.modeling import CausalLMBase
from mini_llm.qwen_model import Qwen3ForCausalLM


MODEL_TYPES: dict[str, type[CausalLMBase]] = {
    "qwen3": Qwen3ForCausalLM,
    "granitemoe": GraniteMoeForCausalLM,
}


def load_model(
    model_dir: str | Path,
    *,
    model_config: DecoderConfig | None = None,
) -> CausalLMBase:
    """Load the model implementation registered for one typed configuration."""

    config = load_config(model_dir) if model_config is None else model_config
    try:
        model_type = MODEL_TYPES[config.model_type]
    except KeyError as exc:
        raise ConfigError(
            f"no model is registered for model_type {config.model_type!r}"
        ) from exc
    return model_type.from_model_dir(model_dir, model_config=config)
