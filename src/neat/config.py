import logging
import sys
import yaml
from util import get_path_of


_default_config = {
    'population_size': 500,
    'generations': 100,
    'input_nodes': 130,
    'output_nodes': 7,

    'create_new_connection_probability': 1.0,
    'create_new_node_probability': 0.2,
    'mutate_new_child_probability': 0.7,
    'mutate_gene_probability': 0.6,
    'change_weight_slightly_probability': 0.9,
    'keep_percent_from_species': 0.7,
    'keep_connection_disabled_probability': 0.75,

    'max_simulation_steps': 20000,
    'parallel': True,
    'num_workers': 3,
    'render': False,
    'log_level': 'INFO'
}
_loaded_config = None


def setdefault_recursively(target, default):
    for key_default in default:
        if isinstance(default[key_default], dict):
            setdefault_recursively(target.setdefault(key_default, {}), default[key_default])
        else:
            target.setdefault(key_default, default[key_default])


def get_config():
    global _loaded_config
    if not _loaded_config:
        _loaded_config = load_config()
    return _loaded_config


def load_config(filename=None):
    if not filename:
        filename = 'neat_params.yml'
    try:
        with open(get_path_of(filename)) as config_file:
            config = yaml.load(config_file)
    except FileNotFoundError:
        config = {}

    setdefault_recursively(config, _default_config)
    global _loaded_config
    if not _loaded_config:
        _loaded_config = config
    return config


def setup_logging(log_level=None):
    if not log_level:
        log_level = _default_config['log_level']
    log_formatter = logging.Formatter('%(asctime)s %(name)s[%(process)d]: %(levelname)s: %(message)s')  # %(module)s:%(funcName)s
    stdout_log_hander = logging.StreamHandler(sys.stdout)
    stdout_log_hander.setFormatter(log_formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.getLevelName(log_level))
    root_logger.addHandler(stdout_log_hander)
