import os
from multiprocessing import Process

if __name__ == "__main__":
    seed = 0
    processes = []
    commands = ["python cs285/scripts/run_experiment.py --which_gpu 1 --data_path ../../BC_HP --expert_policy_file cs285/policies/experts/Hopper.pkl --env_name Hopper-v2 --exp_name FCNN_hp_lr0.005 --n_iter 40 --do_dagger --expert_data cs285/expert_data/expert_data_Hopper-v2.pkl --video_log_freq -1 -lr 0.005 --seed {}".format(str(seed)),
                "python cs285/scripts/run_experiment.py --which_gpu 1 --data_path ../../BC_HP --expert_policy_file cs285/policies/experts/Hopper.pkl --env_name Hopper-v2 --exp_name SIN_hp_lr0.0001 --n_iter 40 --do_dagger --expert_data cs285/expert_data/expert_data_Hopper-v2.pkl --video_log_freq -1 -lr 0.0001 --siren --seed {}".format(str(seed)),
                "python cs285/scripts/run_experiment.py --which_gpu 0 --data_path ../../BC_HP --expert_policy_file cs285/policies/experts/Hopper.pkl --env_name Hopper-v2 --exp_name SIN_gv_hp_lr0.001 --n_iter 40 --do_dagger --expert_data cs285/expert_data/expert_data_Hopper-v2.pkl --video_log_freq -1 -lr 0.001 --siren --supervision_mode gv --train_separate_params --epsilon_s 0.05 --gradient_loss_scale 0.05 --seed {}".format(str(seed)),
                "python cs285/scripts/run_experiment.py --which_gpu 0 --data_path ../../BC_HP --expert_policy_file cs285/policies/experts/Hopper.pkl --env_name Hopper-v2 --exp_name RELU_hp_lr0.0001 --n_iter 40 --do_dagger --expert_data cs285/expert_data/expert_data_Hopper-v2.pkl --video_log_freq -1 -lr 0.0001 --siren --use_relu --seed {}".format(str(seed)),
                "python cs285/scripts/run_experiment.py --which_gpu 0 --data_path ../../BC_HP --expert_policy_file cs285/policies/experts/Hopper.pkl --env_name Hopper-v2 --exp_name RELU_gv_hp_lr0.0005 --n_iter 40 --do_dagger --expert_data cs285/expert_data/expert_data_Hopper-v2.pkl --video_log_freq -1 -lr 0.0005 --siren --use_relu --supervision_mode gv --train_separate_params --epsilon_s 0.05 --gradient_loss_scale 0.05 --seed {}".format(str(seed))]
    for cmd in commands:
        processes.append(Process(target=os.system, args=(cmd,)))
    
    for p in processes:
        p.start()
    for p in processes:
        p.join()


