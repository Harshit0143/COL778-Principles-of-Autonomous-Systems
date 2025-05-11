import os
import torch
import numpy as np
from PIL import Image
from dqn import DQNAgent
from env import get_highway_env, ACTION_NO_OP
from utils import visualize_policy
from tabular_q import TabularQAgent
from tabular_q import validate_policy

        

def visualize_lane_value(agent, visualization_runs, agent_type, lane_visualize_folder_path) -> None:
    os.makedirs(lane_visualize_folder_path, exist_ok = True)
    for episode_idx in range(visualization_runs):
        agent.env.reset(episode_idx)
        episode_finished = False
        k = 0
        while(not episode_finished):
            k += 1
            _, _, episode_finished, _ = agent.env.step(ignore_control_car = True)
            if(k % 20 == 0):
                states = agent.env.get_all_lane_states()
                q_vals = np.zeros((4, 4))
                for state in states:
                    if agent_type == 'tabular':
                        q_value = agent._q_table.get(tuple(state))[ACTION_NO_OP]
                    elif agent_type == 'dqn':
                        with torch.no_grad():
                            q_value = agent._q_fn(torch.tensor(state).unsqueeze(0))[0, ACTION_NO_OP].item()
                    else:
                        print('Incorrect agent type specified, please use tabular or dqn')
                        return 
                    q_vals[state[1], state[0] - 1] = q_value
                image = agent.env.render_lane_state_values(q_vals)
                image_path = os.path.join(lane_visualize_folder_path, f'lane_visualization_{episode_idx:02}_step_{k:04}.png')
                Image.fromarray(image).save(image_path)
    print(f'Lane Visualizations saved to folder [{lane_visualize_folder_path}]')
                



def visualize_speed_value(agent, visualization_runs, agent_type, speed_visualize_folder_path) -> None:
    os.makedirs(speed_visualize_folder_path, exist_ok = True)
    for episode_idx in range(visualization_runs):
        agent.env.reset(episode_idx)
        episode_finished = False
        k = 0
        while(not episode_finished):
            k += 1
            _, _, episode_finished, _ = agent.env.step(ignore_control_car = True)
            
            if(k % 20 == 0):
                states = agent.env.get_all_speed_states()
                qvalues = np.zeros(len(states))
                for state in states:
                    if agent_type == 'tabular':
                        q_value = agent._q_table.get(tuple(state))[ACTION_NO_OP]
                    elif agent_type == 'dqn':
                        with torch.no_grad():
                            q_value = agent._q_fn(torch.tensor(state).unsqueeze(0))[0, ACTION_NO_OP].item()
                    else:
                        print('Incorrect agent type specified, please use tabular or dqn')
                        return
                    qvalues[state[0] - 1] = q_value
                image = agent.env.render_speed_state_values(qvalues)
                image_path = os.path.join(speed_visualize_folder_path, f'speed_visualization_{episode_idx:02}_step_{k:04}.png')
                Image.fromarray(image).save(image_path)
    print(f'Speed Visualizations saved to folder [{speed_visualize_folder_path}]')


def show_validate_policy_result(validate_policy_output):
    print('----------------------------------')
    print(f'Average discounted return: {validate_policy_output[0]:.4f}')
    print(f'Average distance covered : {validate_policy_output[1]:.4f}')


if __name__ == '__main__':
    model_path = 'results/experiments/part1-a/agent/q_tabular_agent.pkl'
    env = get_highway_env(dist_obs_states = 5, reward_type = 'dist', obs_type = 'discrete')
    agent = TabularQAgent(env = env, pretrained = model_path)
    show_validate_policy_result(validate_policy(agent, 1000))
    visualize_policy(agent, 10, 'results/plots/part1-a_gifs/')
    visualize_lane_value(agent, 10, 'tabular', 'results/plots/part1-a-lane_visualizations')
    visualize_speed_value(agent, 10, 'tabular', 'results/plots/part1-a-speed_visualizations')



    env = get_highway_env(dist_obs_states = 5, reward_type = 'overtakes', obs_type = 'discrete')
    model_path = 'results/experiments/part-e.1/agent/q_tabular_agent.pkl'
    agent = TabularQAgent(env = env, pretrained = model_path)
    show_validate_policy_result(validate_policy(agent, 1000))
    visualize_policy(agent, 10, 'results/plots/part1-e.1_gifs/')
    visualize_lane_value(agent, 10, 'tabular', 'results/plots/part1-e.1-lane_visualizations')
    visualize_speed_value(agent, 10, 'tabular', 'results/plots/part1-e.1-speed_visualizations')


    env = get_highway_env(dist_obs_states = 3, reward_type = 'dist', obs_type = 'discrete')
    model_path = 'results/experiments/part-e.2/agent/q_tabular_agent.pkl'
    agent = TabularQAgent(env = env, pretrained = model_path)
    show_validate_policy_result(validate_policy(agent, 1000))
    visualize_policy(agent, 10, 'results/plots/part1-e.2_gifs/')
    visualize_lane_value(agent, 10, 'tabular', 'results/plots/part1-e.2-lane_visualizations')
    visualize_speed_value(agent, 10, 'tabular', 'results/plots/part1-e.2-speed_visualizations')

    

    env = get_highway_env(dist_obs_states = 5, reward_type = 'dist', obs_type = 'discrete')
    model_path = 'results/experiments/part2.1-a/agent/dqn_agent.pkl'
    agent = DQNAgent(env = env, pretrained = model_path)
    show_validate_policy_result(validate_policy(agent, 1000))
    visualize_policy(agent, 10, 'results/plots/part2.1-a_gifs')
    visualize_lane_value(agent, 10, 'dqn', 'results/plots/part2.1-a-lane_visualizations')
    visualize_speed_value(agent, 10, 'dqn', 'results/plots/part2.1-a_speed_visualizations')




    env = get_highway_env(dist_obs_states = 5, reward_type = 'dist', obs_type = 'continuous')
    model_path = 'results/experiments/part2.2-a/agent/dqn_agent.pkl'
    agent = DQNAgent(env = env, pretrained = model_path)
    show_validate_policy_result(validate_policy(agent, 1000))
    visualize_policy(agent, 10, 'results/plots/part2.2-a_gifs')
    visualize_lane_value(agent, 10, 'dqn', 'results/plots/part2.2-a-lane_visualizations')
    visualize_speed_value(agent, 10, 'dqn', 'results/plots/part2.2-a-speed_visualizations')
