import os
import numpy as np
from multiprocessing import Pool


def assemble_mlp_command(base_command, lr, seed, size, exp_name="ant_bc"):
    additional = " --n_layers {} -lr {} --seed {} --data_path ../../mlp_tune_data --exp_name @{}_size{}_lr{}_seed{}@ --no_gpu".format(str(size), str(lr), str(seed), exp_name, str(size), str(lr), str(seed))

    return base_command + additional


def assemble_siren_command(base_command, lr, seed, size, exp_name="ant_bc"):
    additional = " --n_layers {} -lr {} --seed {} --data_path ../../siren_tune_data --exp_name @{}_size{}_lr{}_seed{}@ --siren --no_gpu".format(str(size), str(lr), str(seed), exp_name, str(size), str(lr), str(seed))

    return base_command + additional


if __name__ == "__main__":
    lr_rates = [0.01]
    nums = [2, 3, 4]
    seeds = list(range(6))
    base_command = "python cs285/scripts/run_experiment.py --expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v2 --n_iter 1 --expert_data cs285/expert_data/expert_data_Ant-v2.pkl"
    commands = []

    for lr in lr_rates:
        print("testing mlp lr=", lr)
        for num in nums:
            print("testing mlp layer=", num)
            for s in seeds:
                commands.append(assemble_mlp_command(base_command, lr, s, num ,exp_name="mlp_tune_ant"))

    for lr in lr_rates:
        print("testing siren lr=", lr)
        for num in nums:
            print("testing siren layer=", num)
            for s in seeds:
                commands.append(assemble_siren_command(base_command, lr, s, num, exp_name="siren_tune_ant"))
    pool = Pool(processes=2)
    pool.map(os.system, commands)

