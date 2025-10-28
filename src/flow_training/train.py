import sys
import loguru

logger = loguru.logger
logger.remove()
logger.add(
    sys.stdout,
    level="DEBUG",
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <lvl>{level: <8}</lvl> | {message}",
)

def test_logger():
    logger.info("-" * 50)
    logger.info("train.py called - hello world!")
    logger.info("-" * 50)
