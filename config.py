import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Azure Document Intelligence
AZURE_AFR_API_KEY = os.getenv('AZURE_AFR_API_KEY')
AZURE_AFR_ENDPOINT = os.getenv('AZURE_AFR_ENDPOINT')

# Azure Cognitive Search
AZURE_SEARCH_ENDPOINT = os.getenv('AZURE_SEARCH_ENDPOINT')
AZURE_SEARCH_KEY = os.getenv('AZURE_SEARCH_KEY')
AZURE_INDEX_NAME = os.getenv('AZURE_INDEX_NAME', 'standalone-ece-index')

# LLM Vendor — controls which provider LiteLLM routes calls to.
# Supported values: OPENAI | AZURE | ANTHROPIC | GROQ | OLLAMA
ACTIVE_LLM_VENDOR = os.getenv('ACTIVE_LLM_VENDOR', 'OPENAI')

# Model names — LiteLLM accepts provider-prefixed names (e.g. "groq/llama3-70b-8192")
# or bare names for OpenAI/Azure.  The "azure/" prefix is added automatically for AZURE.
AGENT_CHAT_MODEL = os.getenv('AGENT_CHAT_MODEL', 'gpt-4o-mini')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-ada-002')
VECTOR_SEARCH_DIMENSIONS = int(os.getenv('VECTOR_SEARCH_DIMENSIONS', '1536'))

# OpenAI (ACTIVE_LLM_VENDOR=OPENAI)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Azure OpenAI (ACTIVE_LLM_VENDOR=AZURE)
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_API_VERSION = os.getenv('AZURE_API_VERSION', '2024-02-15-preview')

# Anthropic (ACTIVE_LLM_VENDOR=ANTHROPIC) — set ANTHROPIC_API_KEY in .env
# Groq     (ACTIVE_LLM_VENDOR=GROQ)      — set GROQ_API_KEY in .env
# Ollama   (ACTIVE_LLM_VENDOR=OLLAMA)    — set OLLAMA_API_BASE in .env (no key needed)

# Chunking
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1200'))
BUFFER_SIZE = int(os.getenv('BUFFER_SIZE', '100'))

# Processing
TIMEOUT_ENABLED = os.getenv('TIMEOUT_ENABLED', 'true').lower() == 'true'
AFR_TIMEOUT_SECS = int(os.getenv('AFR_TIMEOUT_SECS', '300'))
MAX_ECE_RETRIALS = int(os.getenv('MAX_ECE_RETRIALS', '3'))

# Server
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', '8001'))

# PostgreSQL (Structured Storage)
POSTGRES_URI = os.getenv('POSTGRES_URI', 'postgresql://user:password@localhost:5432/ece_structured_storage')

# Validate required configs
def validate_config():
    """Validate that all required configuration is present."""
    # Azure infrastructure keys are always required regardless of LLM vendor.
    required_infra = {
        'AZURE_AFR_API_KEY': AZURE_AFR_API_KEY,
        'AZURE_AFR_ENDPOINT': AZURE_AFR_ENDPOINT,
        'AZURE_SEARCH_ENDPOINT': AZURE_SEARCH_ENDPOINT,
        'AZURE_SEARCH_KEY': AZURE_SEARCH_KEY,
        'POSTGRES_URI': POSTGRES_URI,
    }
    missing = [key for key, value in required_infra.items() if not value]

    # LLM API key requirement is vendor-specific.
    vendor = ACTIVE_LLM_VENDOR.upper()
    if vendor == 'OPENAI' and not OPENAI_API_KEY:
        missing.append('OPENAI_API_KEY')
    elif vendor == 'AZURE' and not AZURE_OPENAI_API_KEY:
        missing.append('AZURE_OPENAI_API_KEY')
    # ANTHROPIC / GROQ / OLLAMA keys live in their own env vars and are
    # validated by LiteLLM at the first call; no explicit check here.

    if missing:
        raise ValueError(
            f"Missing required configuration: {', '.join(missing)}. "
            f"Please check your .env file."
        )

# Validate on import
validate_config()
