import os
import time
from datetime import date
from cs285.infrastructure.rl_trainer import RL_Trainer
from cs285.agents.bc_agent import BCAgent
from cs285.policies.loaded_gaussian_policy import LoadedGaussianPolicy
import json
import sys


class BC_Trainer(object):

    def __init__(self, params):

        #######################
        ## AGENT PARAMS
        #######################

        agent_params = {
            'n_layers': params['n_layers'],
            'size': params['size'],
            'learning_rate': params['learning_rate'],
            'max_replay_buffer_size': params['max_replay_buffer_size'],
            'siren': params['siren'],
            'train_separate_params': params['train_separate_params'],
            'supervision_mode': params['supervision_mode'],
            'offset_learning_rate': params['offset_learning_rate'],
            'epsilon_s': params['epsilon_s'],
            'auto_cast': params['auto_cast'],
            'gradient_loss_scale': params['gradient_loss_scale'],
            'additional_activation': params['additional_activation'],
            'omega': params['omega'],
            }

        self.params = params
        self.params['agent_class'] = BCAgent ## HW1: you will modify this
        self.params['agent_params'] = agent_params

        ################
        ## RL TRAINER
        ################

        self.rl_trainer = RL_Trainer(self.params) ## HW1: you will modify this

        #######################
        ## LOAD EXPERT POLICY
        #######################

        print('Loading expert policy from...', self.params['expert_policy_file'])
        self.loaded_expert_policy = LoadedGaussianPolicy(self.params['expert_policy_file'])
        print('Done restoring expert policy...')

    def run_training_loop(self):

        return self.rl_trainer.run_training_loop(
            n_iter=self.params['n_iter'],
            initial_expertdata=self.params['expert_data'],
            collect_policy=self.rl_trainer.agent.actor,
            eval_policy=self.rl_trainer.agent.actor,
            relabel_with_expert=self.params['do_dagger'],
            expert_policy=self.loaded_expert_policy,
        )


def experiment_wrapper(params):
    _experiment(params)


def _experiment(params):


    if params['do_dagger']:
        assert params['n_iter'] > 1, 'DAGGER needs more than 1 iteration (n_iter>1) of training, to iteratively query the expert and train (after 1st warmstarting from behavior cloning).'
    else:
        assert params['n_iter'] == 1, 'Vanilla behavior cloning collects expert data just once (n_iter=1)'

    ## directory for logging
    # date_prefix = date.today().strftime("%m-%d-%Y-")
    # head, tail = os.path.split(params['data_path'])
    # data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), head, date_prefix + tail)
    # if not (os.path.exists(data_path)):
    #     os.makedirs(data_path)
    # log_dir = params['exp_name'] + '_' + time.strftime("%d-%m-%Y_%H-%M-%S") + '_seed-{}'.format(str(params['seed']))
    # log_dir = os.path.join(data_path, log_dir)
    # params['logdir'] = log_dir
    log_dir = params['logdir']
    assert not os.path.exists(log_dir)
    # if not(os.path.exists(log_dir)):
    os.makedirs(log_dir)

    '''Dumping params to file'''
    param_file = open(os.path.join(log_dir, "params.json"), 'w')
    json.dump(params, param_file, indent=2)
    param_file.close()

    '''Redirect STD_OUT'''
    sys.stdout = open(os.path.join(log_dir, "terminal.log"), 'w')


    ###################
    ### RUN TRAINING
    ###################

    trainer = BC_Trainer(params)
    trainer.run_training_loop()
    sys.stdout.close()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_policy_file', '-epf', type=str, required=True)  # relative to where you're running this script from
    parser.add_argument('--expert_data', '-ed', type=str, required=True) #relative to where you're running this script from
    parser.add_argument('--env_name', '-env', type=str, help='choices: Ant-v2, Humanoid-v2, Walker-v2, HalfCheetah-v2, Hopper-v2', required=True)
    parser.add_argument('--exp_name', '-exp', type=str, default='pick an experiment name', required=True)
    parser.add_argument('--do_dagger', action='store_true')
    parser.add_argument('--ep_len', type=int)

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1000)  # number of gradient steps for training policy (per iter in n_iter)
    parser.add_argument('--n_iter', '-n', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=1000)  # training data collected (in the env) during each iteration
    parser.add_argument('--eval_batch_size', type=int,
                        default=1000)  # eval data collected (in the env) for logging metrics
    parser.add_argument('--train_batch_size', type=int,
                        default=100)  # number of sampled data points to be used per gradient/train step

    parser.add_argument('--n_layers', type=int, default=2)  # depth, of policy to be learned
    parser.add_argument('--size', type=int, default=64)  # width of each layer, of policy to be learned
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)  # LR for supervised learning

    parser.add_argument('--video_log_freq', type=int, default=-1)
    parser.add_argument('--scalar_log_freq', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', type=int, default=0)
    parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)
    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--data_path', type=str, default='../../data')
    parser.add_argument('--siren', action='store_true')
    parser.add_argument('--supervision_mode', type=str, default='value')
    parser.add_argument('--epsilon_s', type=float, default=0.05)
    parser.add_argument('--train_separate_params', action='store_true')
    parser.add_argument('--auto_cast', action='store_true')
    parser.add_argument('--offset_learning_rate', type=float, default=5e-3)
    parser.add_argument('--gradient_loss_scale', type=float, default=1.0)
    parser.add_argument('--use_relu', action='store_true')
    parser.add_argument('--additional_activation', type=str, default='identity')


    args = parser.parse_args()

    # convert args to dictionary
    params = vars(args)

    _experiment(params)

if __name__ == "__main__":
    main()
