import sys
import loguru

if __name__ == "__main__":
    logger = loguru.logger
    logger.remove()
    logger.add(
        sys.stdout,
        level="DEBUG",
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <lvl>{level: <8}</lvl> | {message}",
    )

    logger.info("-" * 50)
    logger.info(f"main.py called - hello world!")
    logger.info("-" * 50)
