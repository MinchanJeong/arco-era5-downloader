import sys
import logging
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from utils.paths import LOGGER_DIR

def set_logger(logger_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(logger_path / 'save.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Stream handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    logger.info('Logger initialized.')

    # no need to return logger, as it is a 'root' logger

def set_logger_path(cfg):
# Set Output Path
    y, m, d, H, M, S = pd.Timestamp.now().timetuple()[:6]
    y = str(y)[2:]
    logger_path = Path(LOGGER_DIR, cfg.cfg_name)
    logger_path = Path(logger_path, f"{y}-{m}-{d}", f"{H}-{M}-{S}")

    print(f'>>> Logger Path: {logger_path} <<<')
    logger_path.mkdir(parents=True, exist_ok=True)
    
    OmegaConf.save(cfg.original_cfg, logger_path / f'{cfg.cfg_name}.yaml')

    return logger_path