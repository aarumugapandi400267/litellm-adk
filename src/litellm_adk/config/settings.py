from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    """
    Application settings, loaded from environment variables or .env file.
    """
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore",
        env_prefix="ADK_"  # Support ADK_ prefixed env vars
    )

    # Core LLM Settings
    model: str = Field(default="gpt-4o", description="The model to use.")
    api_key: Optional[str] = Field(default=None, description="Global API key.")
    base_url: Optional[str] = Field(default=None, description="Global base URL.")
    
    # Provider-specific keys (fallback)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    
    # Agent Defaults
    sequential_execution: bool = Field(default=True, description="Default sequential tool execution mode.")

settings = Settings()
