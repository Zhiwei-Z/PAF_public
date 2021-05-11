import os
from multiprocessing import Process

if __name__ == "__main__":
    seed = 0
    processes = []
    commands = ["python cs285/scripts/run_experiment.py --data_path ../../BC_HC --expert_policy_file cs285/policies/experts/HalfCheetah.pkl --env_name HalfCheetah-v2 --exp_name FCNN_hc_lr0.01 --n_iter 40 --do_dagger --expert_data cs285/expert_data/expert_data_HalfCheetah-v2.pkl --video_log_freq -1 --which_gpu 1 -lr 0.01 --seed {}".format(str(seed)),
                "python cs285/scripts/run_experiment.py --data_path ../../BC_HC --expert_policy_file cs285/policies/experts/HalfCheetah.pkl --env_name HalfCheetah-v2 --exp_name SIN_hc_lr0.0001 --n_iter 40 --do_dagger --expert_data cs285/expert_data/expert_data_HalfCheetah-v2.pkl --video_log_freq -1 --which_gpu 1 --siren -lr 0.0001 --seed {}".format(str(seed)),
                "python cs285/scripts/run_experiment.py --data_path ../../BC_HC --expert_policy_file cs285/policies/experts/HalfCheetah.pkl --env_name HalfCheetah-v2 --exp_name SIN_gv_hc_lr0.01 --n_iter 40 --do_dagger --expert_data cs285/expert_data/expert_data_HalfCheetah-v2.pkl --video_log_freq -1  --which_gpu 0 --siren -lr 0.01 --supervision_mode gv --train_separate_params --epsilon_s 0.05 --gradient_loss_scale 0.1 --seed {}".format(str(seed)),
                "python cs285/scripts/run_experiment.py --data_path ../../BC_HC --expert_policy_file cs285/policies/experts/HalfCheetah.pkl --env_name HalfCheetah-v2 --exp_name RELU_hc_lr0.001 --n_iter 40 --do_dagger --expert_data cs285/expert_data/expert_data_HalfCheetah-v2.pkl --video_log_freq -1 --which_gpu 0 --siren --use_relu -lr 0.001 --seed {}".format(str(seed)),
                "python cs285/scripts/run_experiment.py --data_path ../../BC_HC --expert_policy_file cs285/policies/experts/HalfCheetah.pkl --env_name HalfCheetah-v2 --exp_name RELU_gv_hc_lr0.0005 --n_iter 40 --do_dagger --expert_data cs285/expert_data/expert_data_HalfCheetah-v2.pkl --video_log_freq -1  --which_gpu 0 --siren --use_relu -lr 0.0005 --supervision_mode gv --train_separate_params --epsilon_s 0.05 --gradient_loss_scale 0.1 --seed {}".format(str(seed))]
    for cmd in commands:
        processes.append(Process(target=os.system, args=(cmd,)))
    
    for p in processes:
        p.start()
    for p in processes:
        p.join()


