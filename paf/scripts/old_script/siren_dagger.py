import os
import numpy as np
import multiprocessing
from multiprocessing import Pool, Process




def assemble_siren_command(base_command, lr, seed, exp_name, gpu):
    additional = " -lr {} --seed {} --exp_name {} --siren {}".format(str(lr), str(seed), exp_name, gpu)

    return base_command + additional

def assemble_mlp_command(base_command, lr, seed, exp_name, gpu):
    additional = " -lr {} --seed {} --exp_name {} {}".format(str(lr), str(seed), exp_name, gpu)

    return base_command + additional

def assemble_exp_name(env_name="", lr="", dagger="", seed=""):

    return "env{}_lr{}_dagger{}_seed{}".format(env_name, str(lr), str(dagger), str(seed))

"""fetch the policy file for each environment"""
def get_env_policy(env_name):
    if env_name == "ant":
        return "cs285/policies/experts/Ant.pkl"
    if env_name == "hc":
        return "cs285/policies/experts/HalfCheetah.pkl"
    if env_name == "hopper":
        return "cs285/policies/experts/Hopper.pkl"
    if env_name == "humanoid":
        return "cs285/policies/experts/Humanoid.pkl"
    if env_name == "walker":
        return "cs285/policies/experts/Walker2d.pkl"

def get_gpu_command(num):
    rem = num % 5
    if rem == 0:
        return "--no_gpu"
    if rem == 1:
        return "--which_gpu 0"
    if rem == 2:
        return "--which_gpu 0"
    if rem == 3:
        return "--which_gpu 1"
    if rem == 4:
        return "--which_gpu 1"

"""fetch the expert trajectory data for each env"""
def get_env_data(env_name):
    if env_name == "ant":
        return "cs285/expert_data/expert_data_Ant-v2.pkl"
    if env_name == "hc":
        return "cs285/expert_data/expert_data_HalfCheetah-v2.pkl"
    if env_name == "hopper":
        return "cs285/expert_data/expert_data_Hopper-v2.pkl"
    if env_name == "humanoid":
        return "cs285/expert_data/expert_data_Humanoid-v2.pkl"
    if env_name == "walker":
        return "cs285/expert_data/expert_data_Walker2d-v2.pkl"

def get_env_name(env_name):
    if env_name == "ant":
        return "Ant-v2"
    if env_name == "hc":
        return "HalfCheetah-v2"
    if env_name == "hopper":
        return "Hopper-v2"
    if env_name == "humanoid":
        return "Humanoid-v2"
    if env_name == "walker":
        return "Walker2d-v2"


"""
assembles a base command depends on the environment and 
"""
def assemble_base_command(env_name, if_dagger, control_config_string):
    basic = "python3 cs285/scripts/run_experiment.py "
    name_piece = "--env_name {}".format(get_env_name(env_name))
    policy_piece = "--expert_policy_file {}".format(get_env_policy(env_name))
    data_piece = "--expert_data {}".format(get_env_data(env_name))
    dagger_piece = "--do_dagger" if if_dagger else ""
    return "{} {} {} {} {} {}".format(basic, name_piece, policy_piece, data_piece, dagger_piece, control_config_string)



if __name__ == "__main__":
    lr_rates = [0.01, 0.001, 0.0001, 0.00001]
    seeds = list(range(4))
    # control = "--n_iter 50 --n_layers 3 --size 256 --data_path ../../siren_dagger_tune_lr"
    control = "--n_iter 50 --n_layers 3 --size 256 --data_path ../../mlp_dagger_tune_lr --video_log_freq -1"
    gpu = 0
    commands = []
    for env_name in ["ant", "hc", "hopper", "humanoid", "walker"]:
        base_command = assemble_base_command(env_name, if_dagger=True, control_config_string=control)
        for lr in lr_rates:
            print("testing env ", env_name, " lr = ", lr)
            for s in seeds:
                # commands.append(assemble_siren_command(base_command, lr, s, exp_name=assemble_exp_name(env_name, str(lr), "True", str(s)), gpu=get_gpu_command(gpu)))
                commands.append(assemble_mlp_command(base_command, lr, s, exp_name=assemble_exp_name(env_name, str(lr), "True", str(s)), gpu=get_gpu_command(gpu)))
                gpu += 1
    # pool = Pool(processes=3)
    # pool.map(os.system, commands)
    num_process = 5
    i = 0
    while i < len(commands):
        processes = []
        for j in range(i, i + num_process):
            processes.append(Process(target=os.system, args=(commands[j], )) if j < len(commands) else None)
        for p in processes:
            if p:
                p.start()
        for p in processes:
            if p:
                p.join()
        i += num_process




