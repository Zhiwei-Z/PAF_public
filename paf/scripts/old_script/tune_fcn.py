import os
from multiprocessing import Process


def assemble_command(env_name,
                     exp_name,
                     do_dagger=False,
                     num_agent_train_steps_per_iter=1000,
                     n_iter=1,
                     batch_size=1000,
                     eval_batch_size=2000,
                     train_batch_size=100,
                     n_layers=2,
                     size=64,
                     learning_rate=5e-3,
                     video_log_freq=-1,
                     scalar_log_freq=1,
                     which_gpu=0,
                     max_replay_buffer_size=1000000,
                     save_params=False,
                     seed=1,
                     data_path='../../data',
                     siren=False,
                     supervision_mode='value',
                     epsilon_s=0.0,
                     train_separate_params=False,
                     auto_cast=False,
                     offset_learning_rate=5e-3,
                     gradient_loss_scale=1.0,
                     use_relu=False):
    expert_policy_file = '--expert_policy_file {}'.format(get_env_policy(env_name))
    expert_data = '--expert_data {}'.format(get_env_data(env_name))
    env_name = '--env_name {}'.format(get_env_name(env_name))
    exp_name = '--exp_name {}'.format(exp_name)
    do_dagger = '--do_dagger' if do_dagger else ''
    num_agent_train_steps_per_iter = '--num_agent_train_steps_per_iter {}'.format(str(num_agent_train_steps_per_iter))
    n_iter = '--n_iter {}'.format(str(n_iter))
    batch_size = '--batch_size {}'.format(str(batch_size))
    eval_batch_size = '--eval_batch_size {}'.format(str(eval_batch_size))
    train_batch_size = '--train_batch_size {}'.format(str(train_batch_size))
    n_layers = '--n_layers {}'.format(str(n_layers))
    size = '--size {}'.format(str(size))
    learning_rate = '-lr {}'.format(str(learning_rate))
    video_log_freq = '--video_log_freq {}'.format(str(video_log_freq))
    scalar_log_freq = '--scalar_log_freq {}'.format(str(scalar_log_freq))
    which_gpu = '--which_gpu {}'.format(str(which_gpu)) if which_gpu >= 0 else '--no_gpu'
    max_replay_buffer_size = '--max_replay_buffer_size {}'.format(str(max_replay_buffer_size))
    save_params = '--save_params' if save_params else ''
    seed = '--seed {}'.format(str(seed))
    data_path = '--data_path {}'.format(str(data_path))
    siren = '--siren' if siren else ''
    supervision_mode = '--supervision_mode {}'.format(supervision_mode)
    epsilon_s = '--epsilon_s {}'.format(str(epsilon_s))
    train_separate_params = '--train_separate_params' if train_separate_params else ''
    auto_cast = '--auto_cast' if auto_cast else ''
    offset_learning_rate = '--offset_learning_rate {}'.format(str(offset_learning_rate))
    gradient_loss_scale = '--gradient_loss_scale {}'.format(str(gradient_loss_scale))
    use_relu = '--use_relu' if use_relu else ''


    return 'python3 cs285/scripts/run_experiment.py {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}'.format(
        expert_policy_file, expert_data, env_name, exp_name, do_dagger, num_agent_train_steps_per_iter, n_iter,
        batch_size, eval_batch_size, train_batch_size, n_layers, size, learning_rate, video_log_freq, scalar_log_freq,
        which_gpu, max_replay_buffer_size, save_params, seed, data_path, siren, supervision_mode, epsilon_s,
        train_separate_params, auto_cast, offset_learning_rate, gradient_loss_scale, use_relu)


def assemble_exp_name(env_name, var_name, var_value, dagger="", seed=""):
    return "env{}_{}{}_dagger{}_seed{}".format(env_name, var_name, str(var_value), str(dagger), str(seed))


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


def get_gpu_command(num, num_processes):
    rem = num % num_processes
    if rem == 0:
        return 0
    if rem == 1:
        return 0
    if rem == 2:
        return 0
    if rem == 3:
        return 1
    if rem == 4:
        return 1
    return 0


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


if __name__ == "__main__":
    lr_s = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    seeds = list(range(3))
    gpu = 0
    commands = []
    num_process = 7
    for env_name in ["ant", "hc", "hopper", "humanoid", "walker"]:  # ["ant", "hc", "hopper", "humanoid", "walker"]
        for lr in lr_s:
            print("testing env ", env_name, " lr = ", lr)
            for s in seeds:
                commands.append(assemble_command(env_name=env_name,
                                                 n_iter=50,
                                                 video_log_freq=-1,
                                                 which_gpu=get_gpu_command(gpu, num_processes=num_process),
                                                 learning_rate=lr,
                                                 seed=s,
                                                 do_dagger=True,
                                                 use_relu=True,
                                                 exp_name=assemble_exp_name(env_name=env_name, var_name="lr", var_value=lr, dagger='True', seed=s),
                                                 data_path='../../BC_relu_FCN_lr'))
                gpu += 1
    i = 0
    while i < len(commands):
        processes = []
        for j in range(i, i + num_process):
            processes.append(Process(target=os.system, args=(commands[j],)) if j < len(commands) else None)
        for p in processes:
            if p:
                p.start()
        for p in processes:
            if p:
                p.join()
        i += num_process
