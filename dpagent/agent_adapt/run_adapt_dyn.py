import os
import tensorflow as tf
import yaml
import time
import numpy as np
from collections import OrderedDict
from envs.half_cheetah_rand_direc import HalfCheetahRandDirecEnv

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file', type=str, default='halfcheetah_rand_dirc')
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    yaml_path = os.path.abspath('../../yaml/' + args.yaml_file + '.yaml')
    assert (os.path.exists(yaml_path))
    with open(yaml_path, 'r') as f:
        params = yaml.load(f)