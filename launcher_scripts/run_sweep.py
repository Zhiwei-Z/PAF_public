import multiprocessing
import json
from cs285.scripts.run_experiment import experiment_wrapper as run_exp
from cs285.util.cls import EnumEncoder
import os

class Sweeper(object):
    def __init__(self, hyper_config, repeat=1):
        self.hyper_config = hyper_config
        self.hyper_config['seed'] = list(range(repeat))
        assert len(hyper_config['parent_log_dir']) == 1

    def __iter__(self):
        import itertools
        count = 0
        for config in itertools.product(*[val for val in self.hyper_config.values()]):
            kwargs = {key: config[i] for i, key in enumerate(self.hyper_config.keys())}
            kwargs['logdir'] = os.path.join(kwargs['parent_log_dir'], "experiment_" + str(count))
            count += 1
            yield kwargs



def run_sweep(sweep_ops, variant, repeats=2, num_parallel=4):
    ### Need to convert parameters into iterables
    for key in variant.keys():
        variant[key] = [variant[key]]

    for key in sweep_ops.keys():
        variant[key] = sweep_ops[key]

    print(json.dumps(variant, indent=2))
    sweeper = Sweeper(variant, repeat=repeats)
    print("----- Number of parallel threads: ", num_parallel)

    pool = multiprocessing.Pool(num_parallel)
    exp_args = []
    for config in sweeper:
        exp_args.append(config)

    pool.map(run_exp, exp_args)

