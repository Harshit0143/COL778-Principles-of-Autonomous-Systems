import os
from dqn import DQNAgent
from copy import deepcopy
from plotting import plot_logs
from env import get_highway_env
from tabular_q import TabularQAgent


def run_part(AgentClass, default_args, name, vary_exps = True):
    ############################ part a
    print('Running part a..')
    default_args['log_folder'] = f'results/experiments/{name}-a/agent'
    agent = AgentClass(**default_args)
    plot_logs([agent.log_file], save_path = f'results/plots/{name}-a.png')

    if not vary_exps:
        return
    ############################ part b
    print('Running part b..')
    discount_factors = [0.8, 0.9, 0.99]
    vary_experiment(AgentClass, default_args, 'discount_factor', discount_factors, f'{name}-b')
    ############################ part c
    print('Running part c..')
    alphas = [0.1, 0.3, 0.5]
    vary_experiment(AgentClass, default_args, 'alpha', alphas, f'{name}-c')
    ############################ part d
    print('Running part d..')
    eps0 = 1.00
    eps_argss = [
            default_args['eps_args'],
            {'type' : 'exponential', 'init' : eps0, 'itr_eps_half' : 25000},
            {'type' : 'exponential', 'init' : eps0, 'itr_eps_half' : 50000},
            {'type' : 'exponential', 'init' : eps0, 'itr_eps_half' : 100000},
            {'type' : 'exponential', 'init' : eps0, 'itr_eps_half' : 200000},
            {'type' : 'exponential', 'init' : eps0, 'itr_eps_half' : 400000},

            {'type' : 'linear', 'init' : eps0, 'itr_eps_zero' : 25000},
            {'type' : 'linear', 'init' : eps0, 'itr_eps_zero' : 50000},
            {'type' : 'linear', 'init' : eps0, 'itr_eps_zero' : 100000},
            {'type' : 'linear', 'init' : eps0, 'itr_eps_zero' : 150000},
            {'type' : 'linear', 'init' : eps0, 'itr_eps_zero' : 200000},
    ]
    vary_experiment(AgentClass, default_args, 'eps_args', eps_argss, f'{name}-d')


def vary_experiment(AgentClass, default_args, very_arg_name, vary_arg_vals, part_name):
    log_files = []
    for idx, arg_val in enumerate(vary_arg_vals):
        args = deepcopy(default_args)
        args[very_arg_name] = arg_val
        args['log_folder'] = f'results/experiments/{part_name}/agent_{idx}'
        agent = AgentClass(**args)
        log_files.append(agent.log_file)
    plot_logs(log_files, f'results/plots/{part_name}.png')

def run_part_e(default_args):
    print('Running part e..')
    ############################ part e.1
    args = deepcopy(default_args)
    args['env'] = get_highway_env(dist_obs_states = 5, reward_type = 'overtakes', obs_type = 'discrete')
    args['log_folder'] = f'results/experiments/part1-e.1/agent'
    agent = TabularQAgent(**args)
    plot_logs([agent.log_file], save_path = f'results/plots/part1-e.1.png')


    ############################ part e.1
    args = deepcopy(default_args)
    args['env'] = get_highway_env(dist_obs_states = 3, reward_type = 'dist', obs_type = 'discrete')
    args['log_folder'] = f'results/experiments/part1-e.2/agent'
    agent = TabularQAgent(**args)
    plot_logs([agent.log_file], save_path = f'results/plots/part1-e.2.png')

if __name__ == '__main__':
    os.makedirs('results/plots', exist_ok = True)
    default_args =  {
        'env'  : None,
        'alpha': 0.1,
        'eps_args' : {
            'type' : 'constant',
            'init' : 0.75
        },
        'discount_factor' : 0.90,
        'iterations' : 100000,
        'validate_every' : 1000,
        'log_folder' : None,
    }

    default_args['env'] = get_highway_env(dist_obs_states = 5, reward_type = 'dist', obs_type = 'discrete')
    run_part(TabularQAgent, default_args, 'part1')
    run_part_e(default_args)


    default_args['iterations'], default_args['validate_every'] = 200000, 1000
    obs_type = 'discrete'
    default_args['obs_type'] = obs_type
    default_args['env'] = get_highway_env(dist_obs_states = 5, reward_type = 'dist', obs_type = obs_type)
    run_part(DQNAgent, default_args, f'part2.1', vary_exps = False)


    obs_type = 'continuous'
    default_args['obs_type'] = obs_type
    default_args['state_embed_dim'] = 4 
    default_args['env'] = get_highway_env(reward_type = 'dist', obs_type = obs_type)
    run_part(DQNAgent, default_args, f'part2.2', vary_exps = False)
