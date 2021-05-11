import sys
import os
import time
from launcher_scripts import DATA_PATH
import json
from cs285.util.simOptions import get_options
from launcher_scripts.run_sweep import run_sweep
from launcher_scripts.util import config_by_env
from datetime import date


def set_up_parent_log_dir(variant):
    time_prefix = time.strftime("%d-%m-%Y_%H-%M-%S")
    parent_log_dir = os.path.join(DATA_PATH, "{}-{}".format(time_prefix, variant['exp_name']))

    if not (os.path.exists(parent_log_dir)):
        os.makedirs(parent_log_dir)

    variant["parent_log_dir"] = parent_log_dir


def main():
    variant = get_options(sys.argv)

    sweep_ops = {}
    if 'tuning' in variant:
        sweep_ops = json.load(open(variant['tuning'], "r"))

    env_config = config_by_env(variant['env_name'])
    variant.update(env_config)
    set_up_parent_log_dir(variant)

    run_sweep(sweep_ops=sweep_ops,
              variant=variant,
              repeats=variant['random_seeds'],
              num_parallel=variant['num_parallel'])


if __name__ == '__main__':
    main()
