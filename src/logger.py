import logging
import sys


def setup_logging():
    log_formatter = logging.Formatter('%(asctime)s %(name)s[%(process)d]: %(levelname)s: %(message)s')  # %(module)s:%(funcName)s
    stdout_log_hander = logging.StreamHandler(sys.stdout)
    stdout_log_hander.setFormatter(log_formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(stdout_log_hander)
    logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
    logging.getLogger('tensorflow').setLevel(logging.DEBUG)
