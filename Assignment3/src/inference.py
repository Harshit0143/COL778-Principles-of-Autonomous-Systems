import sys
import gym
import torch
from config import configs
from utils.utils import compute_metrics, sample_n_trajectories
from policies.experts import load_expert_policy
from agents.mujoco_agents import ImitationAgent

@torch.no_grad()
def evaluate_agent(env, policy, n_trajs, max_length):
    """
    Evaluate the agent on the environment.
    """
    eval_trajs = sample_n_trajectories(
                                    env = env,
                                    policy = policy,
                                    ntraj = n_trajs,
                                    max_length = max_length,
                                    render = False)
    logs = compute_metrics(trajs = eval_trajs[: 3], eval_trajs = eval_trajs)
    print(f'Mean Episode Length: {logs["Eval_AverageEpLen"]:.5f}')
    print(f'Mean Return        : {logs["Eval_AverageReturn"]:.5f}')
    print(f'Stdev Return       : {logs["Eval_StdReturn"]:.5f}')
    return logs

env_name, n_trajs = sys.argv[1], int(sys.argv[2])
assert env_name in ['Ant-v4', 'Hopper-v4']
env = gym.make(env_name,render_mode = None)
max_ep_len = env.spec.max_episode_steps
obs_dim = env.observation_space.shape[0]
discrete = isinstance(env.action_space, gym.spaces.Discrete)
action_dim = env.action_space.n if discrete else env.action_space.shape[0]



agent = ImitationAgent(
                    observation_dim = obs_dim,
                    action_dim = action_dim,
                    args = None,
                    **configs[env_name]['hyperparameters'])
file_path = 'best_models/Ant-v4.pth'
agent.load_state_dict(torch.load(file_path))
print('Evaluating imitation policy')
print(agent.policy)
evaluate_agent(
            env = env,
            policy = agent.get_action,
            n_trajs = n_trajs,
            max_length = max_ep_len)

print('Evaluating expert policy')
expert_policy, _ = load_expert_policy(env = env, env_name = env_name)
print(expert_policy)
evaluate_agent(
            env = env,
            policy = expert_policy.forward,
            n_trajs = n_trajs,
            max_length = max_ep_len)
