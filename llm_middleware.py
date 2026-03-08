from openai import AsyncOpenAI, AsyncAzureOpenAI
import config


class LLMMiddleware:
    """Standalone LLM middleware for embeddings and chat completions"""
    
    def __init__(self):
        self.openai_api_key = config.OPENAI_API_KEY
        self.azure_openai_api_key = config.AZURE_OPENAI_API_KEY
        self.azure_openai_endpoint = config.AZURE_OPENAI_ENDPOINT
        self.azure_api_version = config.AZURE_API_VERSION
        self.active_vendor = config.ACTIVE_LLM_VENDOR
        self.embedding_model = config.EMBEDDING_MODEL

    def _initialize_openai_client(self):
        """Initialize OpenAI client"""
        return AsyncOpenAI(api_key=self.openai_api_key)

    def _initialize_azure_client(self):
        """Initialize Azure OpenAI client"""
        return AsyncAzureOpenAI(
            api_key=self.azure_openai_api_key,
            azure_endpoint=self.azure_openai_endpoint,
            api_version=self.azure_api_version,
        )

    def initialize_client(self):
        """Initialize client based on active vendor"""
        if self.active_vendor == "OPENAI":
            return self._initialize_openai_client()
        elif self.active_vendor == "AZURE":
            return self._initialize_azure_client()
        else:
            raise ValueError(f"Unknown vendor: {self.active_vendor}")
