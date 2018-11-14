import logging
import os
import sys
import yaml


def setup_logging():
    log_level = 'INFO'  # Default log level
    filename = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                            'simulation_params.yml')
    try:
        with open(filename) as file:
            config = yaml.load(file)
            log_level = config.get('log_level', 'INFO')
    except FileNotFoundError:
        pass

    log_formatter = logging.Formatter('%(asctime)s %(name)s[%(process)d]: %(levelname)s: %(message)s')  # %(module)s:%(funcName)s
    stdout_log_hander = logging.StreamHandler(sys.stdout)
    stdout_log_hander.setFormatter(log_formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.getLevelName(log_level))
    root_logger.addHandler(stdout_log_hander)
    logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
    logging.getLogger('tensorflow').setLevel(logging.DEBUG)
