import logging

def get_custom_logger(logger_name: str = __name__) -> logging.Logger:
    """
    Creates and returns a logger that prints a timestamp,
    logger name, logging level, and message.
    """
    logger = logging.getLogger(logger_name)
    # Prevent adding multiple handlers if the logger already has one
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)  # Set your desired logging level here

        # Create a console handler (stream handler)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # Define formatter to include timestamp, logger name, log level, and message
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
    return logger

logger = get_custom_logger("CustomSplitSkill_FunctionApp")