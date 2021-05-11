import numpy as np
import time
import torch
from cs285.infrastructure import pytorch_util as ptu


############################################
############################################

def sample_trajectory(env, policy, max_path_length, render=False, render_mode=('rgb_array')):

    # initialize env for the beginning of a new rollout
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        if render:  # feel free to ignore this for now
            if 'rgb_array' in render_mode:
                if hasattr(env.unwrapped, 'sim'):
                    if 'track' in env.unwrapped.model.camera_names:
                        image_obs.append(env.unwrapped.sim.render(camera_name='track', height=500, width=500)[::-1])
                    else:
                        image_obs.append(env.unwrapped.sim.render(height=500, width=500)[::-1])
                else:
                    image_obs.append(env.render(mode=render_mode))
            if 'human' in render_mode:
                env.render(mode=render_mode)
                time.sleep(env.model.opt.timestep)
        obs.append(ob)
        ac = policy.get_action(ob)
        ac = ac[0]
        acs.append(ac)
        ob, rew, done, _ = env.step(ac)
        # add the observation after taking a step to next_obs
        next_obs.append(ob)
        rewards.append(rew)
        steps += 1
        # If the episode ended, the corresponding terminal value is 1
        # otherwise, it is 0
        if done or steps > max_path_length:
            terminals.append(1)
            break
        else:
            terminals.append(0)
    return Path(obs, image_obs, acs, rewards, next_obs, terminals)

def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False, render_mode=('rgb_array')):
    """
        Collect rollouts until we have collected min_timesteps_per_batch steps.

        TODO implement this function
        Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
        Hint2: use get_pathlength to count the timesteps collected in each path
    """
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        # collect rollout
        path = sample_trajectory(env, policy, max_path_length, render, render_mode)
        paths.append(path)

        # count steps
        timesteps_this_batch += get_pathlength(path)
        print('At timestep:    ', timesteps_this_batch, '/', min_timesteps_per_batch, end='\r')

    return paths, timesteps_this_batch

def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False, render_mode=('rgb_array')):
    """
        Collect ntraj rollouts.

        TODO implement this function
        Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
    """
    paths = []
    for i in range(ntraj):
        # collect rollout
        path = sample_trajectory(env, policy, max_path_length, render, render_mode)
        paths.append(path)

    return paths

############################################
############################################

def Path(obs, image_obs, acs, rewards, next_obs, terminals, average_gradient=False):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def convert_listofrollouts(paths, concat_rew=True, average_gradients=False, epsilon_s=0):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    gradients = np.concatenate([approximate_traj_grad(path["observation"],
                                                      path['action'],
                                                      average_gradients=average_gradients,
                                                      epsilon_s=epsilon_s) for path in paths])
    if concat_rew:
        rewards = np.concatenate([path["reward"] for path in paths])
    else:
        rewards = [path["reward"] for path in paths]
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    return observations, actions, rewards, next_observations, terminals, gradients

############################################
############################################


def get_pathlength(path):
    return len(path["reward"])


def approximate_traj_grad(observations, actions, average_gradients=False, epsilon_s=0):
    """
    Return the approximated gradients for each observation in the trajectory
    if average_gradients, then for each observation that's no the first or the last,
    we approximate the gradient by taking the average of the forward and backward
    gradients
    """
    length = observations.shape[0]

    # Hard code the first entry
    first_entry = approximate_grad(observations[0], actions[0], observations[1], actions[1], epsilon_s=epsilon_s)
    forward_gradients = np.array([first_entry])

    # Intermediate entries
    for i in range(1, length - 1):
        entry = approximate_grad(observations[i], actions[i], observations[i + 1], actions[i + 1], epsilon_s=epsilon_s)
        forward_gradients = np.append(forward_gradients, [entry], axis=0)

    # Last entry is the same as the second last entry as we have no next_actions
    last_entry = forward_gradients[-1]
    forward_gradients = np.append(forward_gradients, [last_entry], axis=0)

    if not average_gradients:
        return forward_gradients

    # Now compute the backward_gradients
    backward_gradients = forward_gradients
    backward_gradients[1:length - 1] = forward_gradients[0:length - 2]

    average_gradients = (forward_gradients + backward_gradients) / 2
    return average_gradients

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


# def jacobian(y, x):
#     '''
#     Compute the Jacobian matrix in batch form.
#     Return (B, D_y, D_x)
#     '''
#
#     # batch = y.shape[0]
#     # single_y_size = np.prod(y.shape[1:])
#     # y = y.view(batch, -1)
#     # vector = torch.ones(batch).to(y)
#     #
#     # # Compute Jacobian row by row.
#     # # dy_i / dx -> dy / dx
#     # # (B, D) -> (B, 1, D) -> (B, D, D)
#     # jac = [torch.autograd.grad(y[:, i], x,
#     #                            grad_outputs=vector,
#     #                            retain_graph=True,
#     #                            create_graph=True)[0].view(batch, -1)
#     #        for i in range(single_y_size)]
#     # jac = torch.stack(jac, dim=1)
#
#     # return jac
#     return x.grad.data

def jacobian(net, x, ac_dim):
    bs = x.shape[0]
    x = x.squeeze()
    x = x.repeat(ac_dim, 1)
    x.requires_grad_(True)
    y = net(x)
    grad_output = torch.eye(ac_dim).repeat(bs, 1).to(ptu.device)
    grad = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=grad_output, create_graph=True)[0]
    # y.backward(torch.eye(ac_dim).repeat(bs, 1).to(ptu.device), create_graph=True)
    return grad.reshape(bs, ac_dim, x.shape[1])


def approximate_grad(s1, a1, s2, a2, epsilon_s=0):
    """
    Take in two state pairs and return its approximated gradient:
    | da[1]/ds[1]   da[1]/ds[2]   da[1]/ds[3]   ...   |
    | da[2]/ds[1]   da[2]/d[s2]   da[2]/ds[3]   ...   |
    | da[3]/ds[1]   da[2]/d[s2]   da[3]/ds[3]   ...   |
    |     ...           ...           ...             |
    The two states must be continuous in the trajectory
    """
    assert (s1.shape == s2.shape and
            a1.shape == a2.shape)
    da = (a2 - a1).reshape((a1.shape[0], 1))
    ds = (s2 - s1).reshape((1, s1.shape[0]))
    ds[np.where(np.abs(ds) < epsilon_s)] = 0
    ret = da / ds
    ret = np.nan_to_num(ret, nan=0, posinf=0, neginf=0)
    assert (ret.shape == (a1.shape[0], s1.shape[0]))
    return ret
