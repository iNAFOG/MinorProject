from .base import BaseModel, ModelResponse, Action
from .mock import MockModel
from .openai_compatible import OpenAICompatibleModel
from .local import LocalModel


def build_model(config):
    if config.backend == 'mock':
        return MockModel(config)
    if config.backend == 'openai':
        return OpenAICompatibleModel(config)
    if config.backend == 'local':
        return LocalModel(config)
    if config.backend == 'transformers':
        from .transformers_local import TransformersModel  # torch is imported only when selected
        return TransformersModel(config)
    raise ValueError(f'unknown model backend: {config.backend}')
