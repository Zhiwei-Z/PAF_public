from typing import Union

import torch
from torch import nn
from cs285.infrastructure.networks.Siren import Siren, Activation, _str_to_activation

# Activation = Union[str, nn.Module]
#
#
# _str_to_activation = {
#     'relu': nn.ReLU(),
#     'tanh': nn.Tanh(),
#     'leaky_relu': nn.LeakyReLU(),
#     'sigmoid': nn.Sigmoid(),
#     'selu': nn.SELU(),
#     'softplus': nn.Softplus(),
#     'identity': nn.Identity(),
# }

device = None


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
) -> nn.Module:
    """
        Builds a feedforward neural network

        arguments:
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            output_placeholder: the result of a forward pass through the hidden layers + the output layer
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    # TODO: return a MLP. This should be an instance of nn.Module
    # Note: nn.Sequential is an instance of nn.Module.
    layers = []
    layers.append(nn.Linear(input_size, size))
    layers.append(activation)
    for i in range(n_layers):
        layers.append(nn.Linear(size, size))
        layers.append(activation)

    layers.append(nn.Linear(size, output_size))
    layers.append(output_activation)

    return nn.Sequential(*layers)


def build_siren(input_size: int,
                output_size: int,
                n_layers: int,
                size: int,
                additional_activation: Activation = 'identity',
                omega=30):
    return Siren(in_features=input_size,
                 out_features=output_size,
                 hidden_features=size,
                 hidden_layers=n_layers,
                 outermost_linear=True,
                 additional_activation=additional_activation,
                 hidden_omega_0=omega,
                 first_omega_0=omega)


def build_policy(input_size: int,
                 output_size: int,
                 n_layers: int,
                 size: int,
                 activation: Activation = 'tanh',
                 output_activation: Activation = 'identity',
                 siren=False,
                 additional_activation: Activation = 'identity',
                 omega=30):
    if siren:
        return build_siren(input_size=input_size,
                           output_size=output_size,
                           n_layers=n_layers,
                           size=size,
                           additional_activation=additional_activation,
                           omega=omega)
    else:
        return build_mlp(input_size=input_size,
                         output_size=output_size,
                         n_layers=n_layers,
                         size=size,
                         activation=activation,
                         output_activation=output_activation)


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)

def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i + 1]
    return div

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

