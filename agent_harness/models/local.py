from urllib.parse import urlparse
from .openai_compatible import OpenAICompatibleModel


class LocalModel(OpenAICompatibleModel):
    """Inference through an already-running local OpenAI-compatible server."""
    def __init__(self, config):
        if urlparse(config.base_url).hostname not in {'127.0.0.1', 'localhost', '::1'}:
            raise ValueError('LocalModel requires a loopback server')
        super().__init__(config)
