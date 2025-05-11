import argparse
import os
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description = 'Parse command-line arguments.')
    parser.add_argument('--iterations', type = int, required = True, help = 'Number of iterations (integer).')
    parser.add_argument('--validate_every', type = int, default = None, help = 'Run validation every x iterations (integer).')
    parser.add_argument('--validation_runs', type = int, default = 1000, help = 'Number of episodes for validation (integer).')
    parser.add_argument('--visualize_runs', type = int, default = 10, help = 'Number of episodes for visualization (integer).')
    parser.add_argument('--output_folder', type = str, required = True, help = 'Path to the input file.')
    parser.add_argument('--obs_type', type = str, default = 'discrete',  choices = ['discrete', 'continuous'], help = 'observation type')
    args = parser.parse_args()
    if args.validate_every is None:
        args.validate_every = args.iterations
    return args

def show_params(params, log_file):
    with open(log_file, 'w') as f:
        for key in params:
            f.write(f'{key}: {params[key]}\n')
        f.write('\n')

def get_epsilon(eps_args: dict, itr: int) -> float:
    # The first episode is itr = 1, so eps = eps['init'] at itr == 1 
    if eps_args['type'] == 'constant':
        return eps_args['init']
    
    elif eps_args['type'] == 'linear':
        eps_min = eps_args['eps_min'] if 'eps_min' in eps_args else 0
        return max(eps_min, eps_args['init'] * (1 -  (itr - 1) / (eps_args['itr_eps_zero'] - 1)))

    
    elif eps_args['type'] == 'exponential':
        # exponentially decay epsilon of form eps_init * e^(-itr / time_constant)
        eps_min = eps_args['eps_min'] if 'eps_min' in eps_args else 0
        return max(eps_min, eps_args['init'] * 2 ** (-(itr - 1) / (eps_args['itr_eps_half'] - 1)))

    raise NotImplementedError

def run_policy(agent, state):
    """
    Needs support of 
    agent.choose_action(state: List[], greedy =  True)
    agent.env.step(action)
    agent.df
    """
    episode_finished = False
    discounted_reward = 0
    df_t = 1
    while not episode_finished:
        action = agent.choose_action(state, greedy = True)
        s_new, reward, episode_finished, _ = agent.env.step(action)
        state = s_new
        discounted_reward += reward * df_t
        df_t *= agent.df
    return discounted_reward, agent.env.control_car.pos

def validate_policy(agent, validation_runs):
    '''
    Returns:
        tuple of (rewards, dist)
            rewards: average across validation run 
                    discounted returns obtained from first state
            dist: average across validation run 
                    distance covered by control car
    '''
    rewards = []
    dists = []

    for i in range(validation_runs):
        obs = agent.env.reset(i) #don't modify this
        reward, dist = run_policy(agent, obs)
        rewards.append(reward)
        dists.append(dist)

    return sum(rewards) / len(rewards), sum(dists) / len(dists)


def visualize_policy(agent, visualization_runs, gif_folder_path) -> None:
    '''
    Args:
        i: total iterations done so for
    
    Create GIF visulizations of policy for visualization_runs
    '''

    os.makedirs(gif_folder_path, exist_ok = True)


    for episode_idx in range(visualization_runs):
        obs = agent.env.reset(episode_idx) #don't modify this
        episode_finished = False
        images = [agent.env.render()]
        while not episode_finished:
            action = agent.choose_action(obs, greedy = True)
            s_new, _ , episode_finished, _ = agent.env.step(action)
            obs = s_new
            images.append(agent.env.render())


        gif_path = os.path.join(gif_folder_path, f'episode_{episode_idx:02}.gif')
        images = [Image.fromarray(frame) for frame in images]
        images[0].save(
                    gif_path,
                    save_all = True,
                    append_images = images[1:],
                    duration = 200,
                    loop = 1,
                    optimize = True
            )
    print(f'GIF Visualizations saved to folder [{gif_folder_path}]')