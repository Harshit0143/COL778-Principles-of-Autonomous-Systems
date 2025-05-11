import os
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.distributions as distributions
from utils.utils import sample_trajectories


from utils.replay_buffer import ReplayBuffer
import utils.utils as utils
from agents.base_agent import BaseAgent
import utils.pytorch_util as ptu
from policies.experts import load_expert_policy
from utils.pytorch_util import build_mlp



class ImitationAgent(BaseAgent):
    '''
    Please implement an Imitation Learning agent. Read train_agent.py to see how the class is used. 
    
    
    Note: 1) You may explore the files in utils to see what helper functions are available for you.
          2)You can add extra functions or modify existing functions. Dont modify the function signature of __init__ and train_iteration.  
          3) The hyperparameters dictionary contains all the parameters you have set for your agent. You can find the details of parameters in config.py.  
          4) You may use the util functions like utils/pytorch_util/build_mlp to construct your NN. You are also free to write a NN of your own. 
    
    Usage of Expert policy:
        Use self.expert_policy.get_action(observation:torch.Tensor) to get expert action for any given observation. 
        Expert policy expects a CPU tensors. If your input observations are in GPU, then 
        You can explore policies/experts.py to see how this function is implemented.
    '''

    def __init__(self, observation_dim: int,
                action_dim: int,
                args = None,
                discrete: bool = False,
                **hyperparameters):
        
        super().__init__()
        self.hyperparameters = hyperparameters
        self.action_dim  = action_dim
        self.observation_dim = observation_dim
        self.is_action_discrete = discrete
        self.args = args
        self.replay_buffer = ReplayBuffer(50000)
        self.current_best = 0

        self.policy = build_mlp(
            input_size = observation_dim,
            output_size = action_dim,
            n_layers = hyperparameters['n_layers'],
            size = hyperparameters['hidden_size'],
            activation = 'tanh',
            output_activation = 'identity'
        )

        self.optimizer = optim.Adam(self.policy.parameters(), lr = hyperparameters['learning_rate'])
        self.loss_fn = nn.MSELoss()
        self.itr_cnt = 0
        self.best_matric = 0
        print('action_dim        :', self.action_dim)
        print('observation_dim   :', self.observation_dim)
        print('is_action_discrete:', self.is_action_discrete)
        print('hyperparameters   :', self.hyperparameters)


        
    def save_model(self):
        savedir = 'best_models'
        os.makedirs(savedir, exist_ok = True)
        policy_state_dict = self.policy.state_dict()
        policy_state_dict = {f'policy.{k}': v for k, v in policy_state_dict.items()}
        torch.save(policy_state_dict, f'{savedir}/{self.args.env_name}.pth')

    def compute_metrics(self, env, train_trajs):
        max_ep_len = env.spec.max_episode_steps
        eval_trajs, _ = sample_trajectories(
                env = env,
                policy = self.get_action,
                min_timesteps_per_batch = 15 * max_ep_len,
                max_length = max_ep_len,
                render = False
            )
        logs = utils.compute_metrics(trajs = train_trajs, eval_trajs = eval_trajs)
        return logs




   
    def forward(self, observation: torch.FloatTensor):
        '''
        observations: (batch, obs_dim)
        output      : (batch, action_dim) [torch.FloatTensor]
        '''
        return self.policy(observation)


    @torch.no_grad()
    def get_action_blend(self, observation: torch.FloatTensor):
        '''
        observations: (batch, obs_dim)
        output      : (batch, action_dim) [torch.FloatTensor]
        '''
        use_expert =  np.random.uniform(0, 1) < self.hyperparameters['beta']
        return torch.FloatTensor(self.expert_policy.get_action(observation)) if use_expert else self.policy(observation)

    @torch.no_grad()
    def get_action(self, observation: torch.FloatTensor):
        '''
        This is used by train_agent.py for evaluation. So this must return the agent's policy purely.
        observations: (batch, obs_dim)
        output      : (batch, action_dim)   
        '''
        return self.forward(observation)

    
    
    
    def update(self, observations: torch.FloatTensor, actions_expert: torch.FloatTensor):
        '''
        observations: (batch, obs_dim)        
        '''
        actions_pred = self.policy(observations)
        loss_val = self.loss_fn(actions_pred, actions_expert)
        self.optimizer.zero_grad()
        loss_val.backward()
        self.optimizer.step()
        return loss_val.item()




    def train_iteration(self, env, envsteps_so_far, render = False, itr_num = None, **kwargs):
        if not hasattr(self, 'expert_policy'):
            self.expert_policy, initial_expert_data = load_expert_policy(env, self.args.env_name)
            self.replay_buffer.add_rollouts(initial_expert_data) # []
            self.device = ptu.device
        rollouts, total_timesteps = sample_trajectories(
                                    env = env,
                                    policy = self.get_action_blend,
                                    min_timesteps_per_batch = self.hyperparameters['min_timesteps_per_batch'],
                                    max_length =  env.spec.max_episode_steps,
                                    render = render)

        '''
        list of dict: 
        observation      <class 'numpy.ndarray'> (len, obs_dim)
        image_obs        <class 'numpy.ndarray'> (lem, 250, 250, 3) or (0,)
        reward           <class 'numpy.ndarray'> (len,)
        action           <class 'numpy.ndarray'> (len, action_dim)
        next_observation <class 'numpy.ndarray'> (len, obs_dim)
        terminal         <class 'numpy.ndarray'> (lem,)
        '''
        self.replay_buffer.add_rollouts(rollouts)

        print('num_timesteps:', total_timesteps)
        print('num_rollouts :', len(rollouts))
        batch = self.replay_buffer.sample_batch(self.hyperparameters['batch_size'], required = ['obs'])
        batch_obs = torch.FloatTensor(batch['obs'], device = self.device)
        actions_expert =  torch.FloatTensor(self.expert_policy.get_action(batch_obs))
        loss_val = self.update(batch_obs, actions_expert)

        self.itr_cnt += 1
        if self.itr_cnt % self.hyperparameters['save_every'] == 0:
            logs = self.compute_metrics(env = env, train_trajs = rollouts[: 3])
            curr_metric = logs['Eval_AverageReturn'] - logs['Eval_StdReturn']
            print(f'Curr metric        : {curr_metric:.5f}')

            if curr_metric > self.best_matric:
                self.best_matric = curr_metric
                self.save_model()
                print(f'Mean Episode Length: {logs["Eval_AverageEpLen"]:.5f}')
                print(f'Mean Return        : {logs["Eval_AverageReturn"]:.5f}')
                print(f'Stdev Return       : {logs["Eval_StdReturn"]:.5f}')
                print(f'Best metric        : {self.best_matric:.5f}')
                print(f'Best model saved after {self.itr_cnt:04} iterations ')

        return {'episode_loss': loss_val, 'trajectories': rollouts, 'current_train_envsteps': total_timesteps}
      

# TODO: Choice of emtric?

