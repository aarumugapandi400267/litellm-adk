import sys
from loguru import logger
from ..config.settings import settings

def setup_logger():
    """
    Configures loguru logger based on application settings.
    """
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.log_level
    )
    return logger

# Initialize global logger
adk_logger = setup_logger()
adk_logger.info("LiteLLM ADK Logger Initialized")
