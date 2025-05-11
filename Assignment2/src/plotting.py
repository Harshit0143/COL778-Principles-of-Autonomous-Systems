import matplotlib.pyplot as plt
import re


def extract_log_data(log_file, window_size = 5):
    with open(log_file, 'r') as f:
        content = f.readlines()
    
    alpha = None
    epsilon_init = None
    eps_type = None
    eps_param = None
    gamma = None
    episodes = []
    rewards = []
    distances = []
    episodes_eps = []
    epsilons = []
    
    for line in content:
        if line.startswith('alpha:'):
            alpha = float(line.split(':')[1].strip())
        elif line.startswith('df:'):
            gamma = float(line.split(':')[1].strip())
        elif line.startswith('eps_args:'):
            match_init = re.search(r"init': ([0-9.]+)", line)
            match_exp = re.search(r"itr_eps_half': ([0-9]+)", line)
            match_lin = re.search(r"itr_eps_zero': ([0-9]+)", line)
            if match_init:
                epsilon_init = float(match_init.group(1))
            if match_exp:
                eps_type = 'exponential'
                eps_param = int(match_exp.group(1))
            elif match_lin:
                eps_type = 'linear'
                eps_param = int(match_lin.group(1))
        elif 'epsilon:' in line:
            parts = line.split()
            episodes_eps.append(int(parts[1]))
            epsilons.append(float(parts[3]))
        elif line.startswith('episode') and 'rewards' in line:
            parts = line.split()
            episodes.append(int(parts[1]))
            rewards.append(float(parts[3].strip(',')))
            distances.append(float(parts[5]))

    running_return = [sum(rewards[i - window_size : i]) / window_size for i in range(window_size, len(rewards) + 1)]
    running_episoded = episodes[window_size - 1:]
    return {
        'alpha': alpha,
        'epsilon_init': epsilon_init,
        'eps_type': eps_type,
        'eps_param': eps_param,
        'gamma': gamma,
        'episodes': episodes,
        'rewards': rewards,
        'distances': distances,
        'episodes_eps': episodes_eps,
        'epsilons': epsilons,
        'running_returns': running_return,
        'running_episodes': running_episoded,
    }
    

def plot_logs(filenames, save_path):
    data = [extract_log_data(log_file) for log_file in filenames]
        
    base_name = save_path.replace('.png', '') if '.png' in save_path else save_path
    plot_params = [
        {'y_title': 'discounted return from start state (avg)', 'x_vals' : 'episodes', 'y_vals' : 'rewards'},
        {'y_title': 'maximum distance traveled from the start state (avg)', 'x_vals' : 'episodes', 'y_vals' : 'distances'},
        {'y_title': '$\\epsilon$', 'x_vals' : 'episodes_eps', 'y_vals' : 'epsilons'},
        {'y_title': 'running avg of discounted return (avg)', 'x_vals' : 'running_episodes', 'y_vals' : 'running_returns'},

    ]
    for param in plot_params:
        plt.figure(figsize = (8, 5))
        for agent in data:
            episodes = agent[param['x_vals']]
            eps_type = agent['eps_type']
            if eps_type == 'exponential':
                label = f'$\\gamma={agent["gamma"]}, \\alpha={agent["alpha"]}, \\epsilon_0={agent["epsilon_init"]}, \\mathrm{{itr}}_{{0.5}}={agent["eps_param"]}$'
            elif eps_type == 'linear':
                label = f'$\\gamma={agent["gamma"]}, \\alpha={agent["alpha"]}, \\epsilon_0={agent["epsilon_init"]}, \\mathrm{{itr}}_0={agent["eps_param"]}$'
            else:
                label = f'$\\gamma={agent["gamma"]}, \\alpha={agent["alpha"]}, \\epsilon_0={agent["epsilon_init"]}$'
            plt.plot(episodes, agent[param['y_vals']], label = label)
        if len(episodes) == 0:
            print(f'No data for [{param["y_vals"]}] in [{base_name}-{param["y_vals"]}]')
            continue
        plt.xlabel('episodes')
        plt.ylabel(param['y_title'])
        modulo = episodes[-1] // 10
        plt.xticks([idx for idx in episodes if idx % modulo == 0], rotation = 45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{base_name}-{param["y_vals"]}.png', format = 'png', dpi = 500)
        plt.close()

def plot_named_logs(filenames, save_path):
    data = [(extract_log_data(log_file[0]), log_file[1]) for log_file in filenames] 
    base_name = save_path.replace('.png', '') if '.png' in save_path else save_path
    plot_params = [
        {'y_title': 'discounted return from start state (avg)', 'x_vals' : 'episodes', 'y_vals' : 'rewards'},
        {'y_title': 'maximum distance traveled from the start state (avg)', 'x_vals' : 'episodes', 'y_vals' : 'distances'},
        {'y_title': '$\\epsilon$', 'x_vals' : 'episodes_eps', 'y_vals' : 'epsilons'},

    ]
    for param in plot_params:
        plt.figure(figsize = (8, 5))
        for agent, label in data:
            episodes = agent[param['x_vals']]
            plt.plot(episodes, agent[param['y_vals']], label = label)
        plt.xlabel('episodes')
        plt.ylabel(param['y_title'])
        modulo = episodes[-1] // 10
        plt.xticks([idx for idx in episodes if idx % modulo == 0], rotation = 45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{base_name}-{param["y_vals"]}.png', format = 'png', dpi = 500)
        plt.close()