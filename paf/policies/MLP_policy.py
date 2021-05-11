import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim
from cs285.infrastructure.utils import jacobian
import numpy as np
import torch
from torch import distributions
from torch.cuda.amp import GradScaler, autocast
from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure.pytorch_util import Activation
from cs285.policies.base_policy import BasePolicy
from cs285.util.cls import SupervisionMode

class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 supervision_mode: SupervisionMode,
                 discrete=False,
                 learning_rate=1e-5,
                 training=True,
                 nn_baseline=False,
                 siren=False,
                 train_separate_offset=False,
                 offset_learning_rate=None,
                 additional_activation: Activation = 'identity',
                 omega=30,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline
        self.siren = siren
        self.train_separate_offset = train_separate_offset
        self.offset_learning_rate = offset_learning_rate
        self.supervision_mode = supervision_mode
        self.additional_activation = additional_activation
        self.omega = omega
        if self.siren:
            print("Building SIREN ...")

        else:
            print("Building FCN ...")

        if self.train_separate_offset:
            print("Using separate offset parameter ... ")
        if self.additional_activation:
            print("Combining additional activation in the network \"{}\"".format(str(self.additional_activation)))
        print("Training network with {} supervision ... ".format(supervision_mode))

        if self.discrete:
            self.logits_na = ptu.build_policy(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
                siren=siren,
                additional_activation=self.additional_activation,
                activation=self.additional_activation,
                omega=self.omega,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_policy(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
                siren=siren,
                additional_activation=self.additional_activation,
                activation=self.additional_activation,
            )
            self.mean_net.to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )
            self.scaler = GradScaler()

        # initializing offset parameter
        if self.train_separate_offset:
            self.offset = nn.Parameter(
                torch.rand(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.offset_optimizer = optim.Adam([self.offset],
                                               self.offset_learning_rate)

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        observation = ptu.from_numpy(observation)
        action_distribution = self(observation)[0]
        action = action_distribution.sample()  # don't bother with rsample
        if self.train_separate_offset:
            return ptu.to_numpy(action) + ptu.to_numpy(self.offset)
        else:
            return ptu.to_numpy(action)

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor) -> Any:
        # observation = observation.clone().detach().requires_grad_(True)
        if self.discrete:
            logits = self.logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution, observation
        else:
            batch_mean = self.mean_net(observation)
            scale_tril = torch.diag(torch.exp(self.logstd))
            batch_dim = batch_mean.shape[0]
            batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
            action_distribution = distributions.MultivariateNormal(
                batch_mean,
                scale_tril=batch_scale_tril,
            )
            return action_distribution, observation


#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 siren,
                 train_separate_offset=False,
                 supervision_mode: SupervisionMode = SupervisionMode.VALUE_ONLY,
                 offset_learning_rate=1e-5,
                 auto_cast=False,
                 gradient_loss_scale=None,
                 additional_activation: Activation = 'identity',
                 omega=30,
                 **kwargs):
        super().__init__(ac_dim,
                         ob_dim,
                         n_layers,
                         size,
                         siren=siren,
                         train_separate_offset=train_separate_offset,
                         offset_learning_rate=offset_learning_rate,
                         supervision_mode=supervision_mode,
                         additional_activation=additional_activation,
                         omega=omega,
                         **kwargs)
        self.auto_cast = auto_cast
        self.gradient_loss_scale = gradient_loss_scale
        if self.auto_cast:
            print("Applying Autocast")
        else:
            print("No Autocast")
        self.loss = nn.MSELoss()
        if self.train_separate_offset:
            self.offset_loss = nn.MSELoss()

    def compute_loss(self, observations, gradients, actions):
        # if self.siren:
        if self.supervision_mode in ['gradient', 'gv']:
            def net(x):
                action_distribution, obs = self(x)
                return action_distribution.rsample()

            prediction_gradients = jacobian(net=net, x=observations, ac_dim=self.ac_dim)
            if self.supervision_mode == 'gradient':
                loss = self.loss(prediction_gradients, ptu.from_numpy(gradients))
            else:  # supervision_mode= 'gv', weight gradient loss
                action_value_loss = nn.MSELoss()
                predicted_actions = self(observations)[0].rsample()
                loss = self.gradient_loss_scale * self.loss(prediction_gradients, ptu.from_numpy(gradients)) + action_value_loss(
                    predicted_actions, ptu.from_numpy(actions))
        else:
            assert self.supervision_mode == 'value'
            predicted_actions = self(observations)[0].rsample()
            loss = self.loss(predicted_actions, ptu.from_numpy(actions))
        return loss

    def update(
            self, observations, actions, gradients=None,
            adv_n=None, acs_labels_na=None, qvals=None
    ):
        observations = ptu.from_numpy(observations)
        # zero grad
        self.optimizer.zero_grad()
        if self.auto_cast:
            # compute loss
            with autocast():
                loss = self.compute_loss(observations=observations, actions=actions, gradients=gradients)
            # scale, backward, update
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # compute loss
            loss = self.compute_loss(observations=observations, actions=actions, gradients=gradients)
            # update
            loss.backward()
            self.optimizer.step()

        # simple training for offset
        if self.train_separate_offset:
            predicted_actions = self(observations)[0].sample()  # remove gradient flow
            predicted_actions = predicted_actions + self.offset
            offset_loss = self.offset_loss(predicted_actions, ptu.from_numpy(actions))
            self.offset_optimizer.zero_grad()
            offset_loss.backward()
            self.offset_optimizer.step()
        else:
            offset_loss = torch.zeros(1)

        return {
            'Training Loss': ptu.to_numpy(loss),
            'Offset_loss': ptu.to_numpy(offset_loss),
        }