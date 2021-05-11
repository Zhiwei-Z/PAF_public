import os
import numpy as np
from multiprocessing import Pool


def assemble_mlp_command(base_command, lr, seed, exp_name="ant_bc"):
    additional = " -lr {} --seed {} --data_path ../../mlp_tune_data --exp_name @{}_lr{}_seed{}@ --no_gpu".format(str(lr), str(seed), exp_name, str(lr), str(seed))

    return base_command + additional


def assemble_siren_command(base_command, lr, seed, exp_name="ant_bc"):
    additional = " -lr {} --seed {} --data_path ../../siren_tune_data --exp_name @{}_lr{}_seed{}@ --siren --no_gpu".format(str(lr), str(seed), exp_name, str(lr), str(seed))

    return base_command + additional


if __name__ == "__main__":
    lr_rates = np.linspace(0.005, 0.05, num=10)
    seeds = list(range(6))
    base_command = "python cs285/scripts/run_experiment.py --expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v2 --n_iter 1 --expert_data cs285/expert_data/expert_data_Ant-v2.pkl"
    commands = []
    for lr in lr_rates:
        print("testing mlp lr=", lr)
        for s in seeds:
            commands.append(assemble_mlp_command(base_command, lr, s, exp_name="siren_tune_ant"))

    for lr in lr_rates:
        print("testing siren lr=", lr)
        for s in seeds:
            commands.append(assemble_siren_command(base_command, lr, s, exp_name="siren_tune_ant"))
    pool = Pool(processes=2)
    pool.map(os.system, commands)

