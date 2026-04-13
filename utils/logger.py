import logging
import os
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from config import config


def setup_logger():
    log_dir = config.General.LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"xsec_{datetime.now().strftime('%Y-%m-%d')}.log")

    logger = logging.getLogger("xsec_alpha")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        return logger

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = TimedRotatingFileHandler(log_file, when="midnight", interval=1, backupCount=30)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


logger = setup_logger()
