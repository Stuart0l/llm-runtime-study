from __future__ import annotations

from unittest.mock import MagicMock, patch
import unittest

from mini_llm.config import ConfigError, DecoderConfig
from mini_llm.model_loader import MODEL_TYPES, load_model
from tests.test_qwen_model import _tiny_config


class ModelLoaderTests(unittest.TestCase):
    def test_dispatches_with_the_already_loaded_configuration(self) -> None:
        config = _tiny_config()
        model_type = MagicMock()
        model = MagicMock()
        model_type.from_model_dir.return_value = model

        with patch.dict(MODEL_TYPES, {"qwen3": model_type}, clear=True):
            actual = load_model("model", model_config=config)

        self.assertIs(actual, model)
        model_type.from_model_dir.assert_called_once_with(
            "model", model_config=config
        )

    def test_rejects_an_unregistered_model_type(self) -> None:
        config = MagicMock(spec=DecoderConfig)
        config.model_type = "unknown"

        with (
            patch.dict(MODEL_TYPES, {}, clear=True),
            self.assertRaisesRegex(ConfigError, "no model is registered"),
        ):
            load_model("model", model_config=config)


if __name__ == "__main__":
    unittest.main()
