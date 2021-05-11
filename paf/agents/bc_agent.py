from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.policies.MLP_policy import MLPPolicySL
from .base_agent import BaseAgent


class BCAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(BCAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params

        # actor/policy
        self.actor = MLPPolicySL(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            siren=self.agent_params['siren'],
            train_separate_offset=self.agent_params['train_separate_params'],
            supervision_mode=self.agent_params['supervision_mode'],
            offset_learning_rate=self.agent_params['offset_learning_rate'],
            auto_cast=self.agent_params['auto_cast'],
            gradient_loss_scale=self.agent_params['gradient_loss_scale'],
            additional_activation=self.agent_params['additional_activation'],
            omega=self.agent_params['omega']
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(self.agent_params['max_replay_buffer_size'],
                                          epsilon_s=self.agent_params['epsilon_s'])

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n, gradients):
        # training a BC agent refers to updating its actor using
        # the given observations and corresponding action labels
        log = self.actor.update(ob_no, ac_na, gradients=gradients)  # HW1: you will modify this
        return log

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(batch_size)  # HW1: you will modify this

    def save(self, path):
        return self.actor.save(path)