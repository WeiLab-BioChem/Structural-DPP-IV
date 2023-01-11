from config.load_constant import constant
from util import util_file


def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    new_dict = {}
    for dict_i in dict_args:
        new_dict.update(dict_i)
    return new_dict


def load_default_args_dict(type_):
    # train
    if type_ == 'StructuralDPPIV':
        config_Script = util_file.read_yaml_to_dict(constant['path_settings'] + 'StructuralDPPIV.yaml')
    else:
        config_Script = util_file.read_yaml_to_dict(constant['path_settings'] + 'Script.yaml')
    config_Lightning = util_file.read_yaml_to_dict(constant['path_settings'] + 'Lightning.yaml')
    config_Model = util_file.read_yaml_to_dict(constant['path_hparams'] + config_Script['model_hparams'])
    args_dict = merge_dicts(config_Script, config_Lightning, config_Model)
    return args_dict
