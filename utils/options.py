import argparse
import random
import yaml
from collections import OrderedDict
from os import path as osp

import torch

from .misc import make_exp_dirs, set_random_seed
from .dist_util import get_dist_info, init_dist


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def merge_configs(base, override):
    """Recursively merges override into base."""
    if not override:
        return base  # No changes if override is None
    for key, value in override.items():
        if isinstance(value, dict) and key in base:
            merge_configs(base[key], value)
        else:
            base[key] = value
    return base

def parse(opt_path, root_path, is_train=True, override_path=None):
    """Parse option file.

    Args:
        opt_path (str): Option file path.
        root_path (str): Root path.
        is_train (str): Indicate whether in training or not. Default True.

    Returns:
        (dict): Options.
    """
    # read config yaml file
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)
        # set name to to the name of config file
        opt['name'] = osp.splitext(osp.basename(opt_path))[0]
    if override_path:
        with open(override_path, mode='r') as f:
            Loader, _ = ordered_yaml()
            override = yaml.load(f, Loader=Loader)
            opt = merge_configs(opt, override)
            # append name of override to the name of config file
            opt['name'] = opt['name'] + '_' + osp.splitext(osp.basename(override_path))[0]
    opt['is_train'] = is_train

    # set number of gpus
    if opt['num_gpu'] == 'auto':
        opt['num_gpu'] = torch.cuda.device_count()

    # paths
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)

    sub_folder = opt_path.split('/')[1]
    if is_train:  # specify training log paths
        experiments_root = osp.join(root_path, 'experiments', sub_folder, opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['log'] = osp.join(experiments_root, 'log')
    else:  # specify test log paths
        if 'test_dataset' in opt['datasets'] and 'bending_weight' in opt['datasets']['test_dataset']:
            bending_weight = opt['datasets']['test_dataset']['bending_weight']
        else:
            bending_weight = ''
        results_root = osp.join(root_path, 'results', sub_folder, opt['name']+str(opt['datasets']['test_dataset']['num_evecs'])+'_'+str(bending_weight))
        opt['path']['results_root'] = results_root
        opt['path']['log'] = osp.join(results_root, 'log')
        opt['path']['visualization'] = osp.join(results_root, 'visualization')
        opt['path']['pcks'] = osp.join(results_root, 'pcks')
        opt['path']['heatmaps'] = osp.join(results_root, 'heatmaps')
    
    return opt


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + str(k) + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        elif isinstance(v, list):
            msg = ''
            for iv in v:
                if isinstance(iv, dict):
                    msg += dict2str(iv, indent_level)
                else:
                    msg += '\n' + ' ' * (indent_level * 2) + str(iv)
        else:
            msg += ' ' * (indent_level * 2) + str(k) + ': ' + str(v) + '\n'
    return msg


def parse_options(root_path, is_train=True, override_path=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument("--override", type=str, default=None, help="Optional override YAML")
    args = parser.parse_args()
    opt = parse(args.opt, root_path, is_train=is_train, override_path=args.override)

    # distributed settings
    if opt['backend'] == 'dp':
        opt['dist'] = False
        print('Backend DataParallel.', flush=True)
    elif opt['backend'] == 'ddp':
        opt['dist'] = True
        port = opt.get('port', 29500)
        init_dist(port=port)
        print('Backend DistributedDataParallel.', flush=True)
    else:
        raise ValueError(f'Invalid backend option: {opt["backend"]}, only supports "dp" and "ddp"')

    # set rank and world_size
    opt['rank'], opt['world_size'] = get_dist_info()

    # make experiment directories
    make_exp_dirs(opt)

    # set random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    return opt
