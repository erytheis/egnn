import logging
import logging.config
import os
from pathlib import Path


def setup_logging(log_dir, log_config='logger/logger_config.json', default_level=logging.INFO):
    """
    Setup logging configuration
    """
    logging.root.setLevel(logging.NOTSET)

    log_config = Path(log_config)
    # if log_config.is_file():
    #     config = read_json(log_config)
    #     # modify logging paths based on run config
    #     for _, handler in config['handlers'].items():
    #         if 'filename' in handler:
    #             handler['filename'] = str(save_dir / handler['filename'])
    #
    #     logging.config.dictConfig(config)
    # else:
    print("Warning: logging configuration file is not found in {}.".format(log_config))
    logging.basicConfig(level=default_level, handlers=[
        logging.FileHandler(os.path.join(log_dir, "run.log")),
        logging.StreamHandler()
    ])  # , filename=save_dir / 'log.log')


class Logger:
    def __init__(self):
        pass
