
import os
import pickle
import numpy as np 
from itertools import product
from env import get_highway_env
from utils import show_params, get_epsilon, validate_policy



class BestAgent:
    
    def __init__(self, iterations = None):


        self.env = get_highway_env(dist_obs_states = 5, reward_type = 'dist', obs_type = 'discrete')
        self.df = 0.9
        log_folder = 'results/tabular_q_best'
        pretrained = None
        self.itrs = 0
        
        self._q_table = self.__get_empty_q_table()
        if pretrained is not None:
            metadata = pickle.load(open(pretrained, 'rb'))
            self.df = metadata['df']
            self._q_table = metadata['_q_table']
            self.__make_policy()
            print(validate_policy(self, 1000))
            # return

        self.log_file = os.path.join(log_folder, 'train_logs.txt')
        self.log_folder = log_folder
        os.makedirs(log_folder, exist_ok = True)
        params = {
            'df': self.df,
            'alpha': 0.1,
            'eps_args': {'type' : 'linear', 'init' : 1.0, 'itr_eps_zero' : 50000, 'eps_min': 0.0},
            'validation_runs': 1000,
            'validate_every': 10000,
        }
        show_params(params, self.log_file)
        self._eval = False
        self.__train_policy(params, 100000)
        params['alpha'] = 0.01
        self.__train_policy(params, 30000)
        self.save_model(130000)


    def save_model(self, itr):
        savepath = os.path.join(self.log_folder, f'q_tabular_agent-{itr:06}.pkl')
        with open(savepath, 'wb') as f:
            pickle.dump({
                'df': self.df,
                '_q_table': self._q_table,
            }, f)
        with open(self.log_file, 'a') as f:
            f.write(f'Model saved to {savepath} after {itr} iterations\n')


    def __get_empty_q_table(self):
        keys = product({1, 2, 3, 4}, range(4), range(5), range(5), range(5), range(5))
        return {key: [0, 0, 0, 0, 0] for key in keys}
    
    def __make_policy(self):
        self._eval = True
        self._policy = {key: np.argmax(val) for key, val in self._q_table.items()}
    
    def __train_policy(self, params, num_itr):
        '''
        Learns the policy
        '''
        #TO DO: You can add you code here
        print(f'Learning policy for {num_itr} iterations...' )
        for episode_idx in range(1, num_itr + 1):
            self.itrs += 1
            eps = get_epsilon(params['eps_args'], episode_idx)
            state = self.env.reset(episode_idx)
            episode_finished = False
            while not episode_finished:
                action = self.choose_action(state, greedy = np.random.rand() > eps)
                s_new, reward, episode_finished, _ = self.env.step(action)
                
                if episode_finished:
                    target = reward
                else:
                    target = reward + self.df * max(self._q_table[tuple(s_new)])
            
                self._q_table[tuple(state)][action] += params['alpha'] * (target - self._q_table[tuple(state)][action])
                state = s_new
           
            with open(self.log_file, 'a') as f:
                f.write(f'episode {self.itrs} epsilon: {eps:.5f}\n') 
                f.write(f"episode {self.itrs} alpha  : {params['alpha']:.5f}\n")

            if episode_idx % params['validate_every'] == 0:
                self.__make_policy()
                rewards_avg, dist_avg = validate_policy(self, validation_runs = params['validation_runs'])
                print(f'episode {self.itrs} rewards: {rewards_avg:.5f}, dist: {dist_avg:.5f}') 
                with open(self.log_file, 'a') as f:
                    f.write(f'episode {self.itrs} rewards: {rewards_avg:.5f}, dist: {dist_avg:.5f}\n') 
                self._eval = False
                
    def get_policy(self):
        return

    def choose_action(self, state, greedy = True):
        '''
        Right now returning random action but need to add
        your own logic
        '''
        if not greedy:
            return np.random.randint(0, 5)
        if self._eval:
            return self._policy[tuple(state)]
        return np.argmax(self._q_table[tuple(state)])

   
if __name__ == '__main__':
    agent = BestAgent()




# SEED = 42
# from env import HighwayEnv, ACTION_NO_OP, get_highway_env, EPISODE_STEPS
# import numpy as np 
# from typing import Tuple
# import torch
# from torch import nn
# import random
# import pickle
# torch.manual_seed(SEED)
# random.seed(SEED)  
# from collections import OrderedDict, deque
# from utils import get_epsilon, show_params
# import argparse
# from itertools import product
# from time import time
# import os

# '''
# import _ Agent
# agent = Agent()
# env = Env()
# agent.train_policy(iterations = 1000)
 
# '''


# def run_policy(agent, state):
#     """
#     Needs support of 
#     agent.choose_action(state: List[], greedy =  True)
#     agent.env.step(action)
#     agent.df
#     """
#     episode_finished = False
#     discounted_reward = 0
#     df_t = 1
#     num_steps = 0
#     state.append(num_steps / EPISODE_STEPS)
#     while not episode_finished:
#         action = agent.choose_action(state, greedy = True)
#         s_new, reward, episode_finished, _ = agent.env.step(action)
#         num_steps += 1
#         s_new.append(num_steps / EPISODE_STEPS)
#         state = s_new
#         discounted_reward += reward * df_t
#         df_t *= agent.df
#     return discounted_reward, agent.env.control_car.pos

# def validate_policy(agent, validation_runs) -> Tuple[float, float]:
#     '''
#     Returns:
#         tuple of (rewards, dist)
#             rewards: average across validation run 
#                     discounted returns obtained from first state
#             dist: average across validation run 
#                     distance covered by control car
#     '''
#     rewards = []
#     dists = []

#     for i in range(validation_runs):
#         obs = agent.env.reset(i) #don't modify this
#         reward, dist = run_policy(agent, obs)
#         rewards.append(reward)
#         dists.append(dist)

#     return sum(rewards) / len(rewards), sum(dists) / len(dists)

# class QFunction(nn.Module):
#     def __init__(self, state_embed_dim, hidden_dim, mode):
#         super(QFunction, self).__init__()
#         num_speed, num_lane, num_dist = 4, 4, 5
#         action_dim = 5
#         if mode == 'discrete':
#             state_dim = num_speed * num_lane * (num_dist ** 4)
#             self.weights = torch.tensor([
#                 num_lane * (num_dist ** 4),
#                 num_dist ** 4,
#                 num_dist ** 3,
#                 num_dist ** 2,
#                 num_dist ** 1,
#                 1
#                 ], dtype = torch.int64)

#             self.encode_fn = self.encode_discrete
#             ff_input_dim = state_embed_dim

#         elif mode == 'continuous':
#             state_dim = num_speed * num_lane
#             self.weights = torch.tensor([
#                 num_lane,
#                 1
#             ], dtype = torch.int64)
#             self.encode_fn = self.encode_continious
#             ff_input_dim = state_embed_dim + 5
        
#         else:
#             raise(NotImplementedError(f'obs_type {mode} not implemented'))
        
#         self.encode_layer = nn.Embedding(num_embeddings = state_dim,  embedding_dim = state_embed_dim)

     
#         # self.layers = nn.Sequential(OrderedDict([
#         #     ('input  ', nn.Linear(ff_input_dim, hidden_dim)),
#         #     ('relu1  ', nn.ReLU()),
#         #     ('hidden1', nn.Linear(hidden_dim, hidden_dim)),
#         #     ('relu2  ', nn.ReLU()),
#         #     ('hidden2', nn.Linear(hidden_dim, hidden_dim)),
#         #     ('relu3  ', nn.ReLU()),
#         #     ('output ', nn.Linear(hidden_dim, action_dim))
#         # ]))
             
#         self.layers = nn.Sequential(OrderedDict([
#             ('input  ', nn.Linear(ff_input_dim, 64)),
#             ('relui  ', nn.ReLU()),
#             ('hidden1', nn.Linear(64, 128)),
#             ('relu1  ', nn.ReLU()),
#             ('hidden2', nn.Linear(128, 128)),
#             ('relu2  ', nn.ReLU()),
#             ('hidden3', nn.Linear(128, 64)),
#             ('relu3  ', nn.ReLU()),
#             ('output ', nn.Linear(64, action_dim))
#         ]))


#     def encode_continious(self, state):
#         state[: , 0] -= 1
#         return torch.cat((self.encode_layer(torch.matmul(state[:, : 2].long(), self.weights)), state[:, 2: ]), dim = 1)

#     def encode_discrete(self, state):
#         state[:, 0] -= 1
#         return self.encode_layer(torch.matmul(state, self.weights))

        
#     def forward(self, state):
#         state = self.encode_fn(state)
#         return self.layers(state)
    



# class BestAgent:
#     def __init__(self, 
#                     env: HighwayEnv, 
#                     eps_args: dict = {'type': 'constant', 'init': 0.75},
#                     alpha: float = 0.1,  
#                     discount_factor: float = 0.9,
#                     iterations: int = 10000, 
#                     log_folder: str = 'results/experiments/dqn',
#                     obs_type: str = 'discrete',
#                     lr: float = 0.0001,
#                     tau: float = 0.005,
#                     batch_size: int = 256,
#                     buffer_size: int = 10000,
#                     hidden_dim: int = 32, 
#                     validation_runs: int = 1000,
#                     validate_every: int = 10000,
#                     visualize_runs: int = 10, 
#                     visualize_every: int = 50000,
#                     state_embed_dim: int = 16,
#                     pretrained: str = None,
#                     use_ddqn: bool = False,
#     ):

#         keys = product({1, 2, 3, 4}, range(4), range(5), range(5), range(5), range(5))
#         self.states = [key for key in keys]

#         self.env = env 
#         if pretrained is not None:
#             metadata = pickle.load(open(pretrained, 'rb'))
#             self.df = metadata['df']
#             self._q_fn = QFunction(state_embed_dim = metadata['state_embed_dim'],
#                                 hidden_dim = metadata['hidden_dim'], 
#                                 mode = metadata['obs_type'])
#             self._q_fn.load_state_dict(torch.load(metadata['state_dict_path']))

#             if metadata['obs_type'] == 'discrete':
#                 self.__make_policy_discrete()
#             else:
#                 self._use_policy = False
#             return


#         self.df = discount_factor
        
#         self.log_file = os.path.join(log_folder, f'train_logs.txt')
#         os.makedirs(log_folder, exist_ok = True)
#         self.obs_type = obs_type

#         params = {
#             'df': self.df,
#             'obs_type': self.obs_type,
#             'iterations': iterations,
#             'alpha': alpha,
#             'eps_args': eps_args,
#             'validation_runs': validation_runs,
#             'validate_every': validate_every,
#             'visualize_runs': visualize_runs,
#             'visualize_every': visualize_every,
#             'state_embed_dim': state_embed_dim,
#             'hidden_dim': hidden_dim,
#             'lr': lr,
#             'batch_size': batch_size,
#             'buffer_size': buffer_size,
#             'use_ddqn': use_ddqn,
#             'tau': tau,
#             'weight_decay': 0.01,
#         }
#         show_params(params, self.log_file)

#         self._q_fn = QFunction(state_embed_dim = state_embed_dim, hidden_dim = hidden_dim, mode = obs_type) 

#         _t_fn = None
#         if params['use_ddqn']:
#             _t_fn = QFunction(state_embed_dim = state_embed_dim, hidden_dim = hidden_dim, mode = obs_type) 
#             _t_fn.load_state_dict(self._q_fn.state_dict())
#         # encoding = self._q_fn.encode_fn(torch.tensor(self.states))
#         # assert (encoding[1 : ] - encoding[: -1]).min() > 0
                
#         self._use_policy = False
#         self.__train_policy(params, _t_fn)

#         savepath = os.path.join(log_folder, 'dqn_agent')
#         state_dict_savepath = f'{savepath}.pth'
#         metadata = {
#             'df': self.df,
#             'obs_type': self.obs_type,
#             'state_embed_dim': state_embed_dim,
#             'hidden_dim': hidden_dim,
#             'state_dict_path': state_dict_savepath
#         }
#         torch.save(self._q_fn.state_dict(), state_dict_savepath)
#         with open(f'{savepath}.pkl', 'wb') as f:
#             pickle.dump(metadata, f)
#         with open(self.log_file, 'a') as f:
#             f.write(f'metadatal saved to {savepath}.pkl after {params["iterations"]} iterations\n')



#     def __make_policy_discrete(self):
#         self._use_policy = True
#         with torch.no_grad():
#             output = self._q_fn(torch.tensor(self.states)).max(dim = 1).indices
#         self._policy = {}
#         for idx in range(len(self.states)):
#             self._policy[tuple(self.states[idx])] = output[idx].item()



#     def __train_policy(self, params, _t_fn):
#         '''
#         Learns the policy
#         '''
#         print(f'Learning policy for {params["iterations"]} iterations...' )
#         # loss_fn =  nn.SmoothL1Loss()
#         loss_fn = nn.MSELoss()
#         optimizer = torch.optim.AdamW(self._q_fn.parameters(), lr = params['lr'], amsgrad = True)#, weight_decay = params['weight_decay'])

#         # optimizer = torch.optim.Adam(self._q_fn.parameters(), lr = params['lr'])
#         # optimizer = torch.optim.Adam(self._q_fn.parameters(), lr = params['lr'])#, amsgrad = True, weight_decay = params['weight_decay'])
#         buffer = deque(maxlen = params['buffer_size'])
#         t_start = time()
#         for episode_idx in range(1, params['iterations'] + 1):
#             eps = get_epsilon(params['eps_args'], episode_idx)
#             episode_finished = False
            
#             state = self.env.reset(episode_idx)
#             num_steps = 0
#             state.append(num_steps / EPISODE_STEPS)
#             while not episode_finished:
#                 action = self.choose_action(state, greedy = np.random.rand() > eps)
#                 s_new, reward, episode_finished, _ = self.env.step(action)
#                 num_steps += 1
#                 s_new.append(num_steps / EPISODE_STEPS)

#                 if reward < 0:
#                     reward *= (1 -  (num_steps - 1) / EPISODE_STEPS)
#                 if reward > 0:
#                     reward += 0.15 * num_steps / EPISODE_STEPS
#                 buffer.append((state, action, s_new, reward, int(not episode_finished)))

#                 state = s_new

#             batch = random.sample(buffer, min(len(buffer), params['batch_size']))
#             s0_batch, action_batch, s1_batch, rewards_batch, run_batch = zip(*batch)
#             s0_batch, action_batch, s1_batch, rewards_batch, run_batch = torch.tensor(s0_batch), torch.tensor(action_batch), torch.tensor(s1_batch), torch.tensor(rewards_batch), torch.tensor(run_batch)
#             pred = self._q_fn(s0_batch).gather(dim = 1, index = action_batch.unsqueeze(1)).squeeze(1)

#             with torch.no_grad():
#                 if params['use_ddqn']:
#                     target = self.df * _t_fn(s1_batch).max(dim = 1).values * run_batch + rewards_batch
#                 else:
#                     target = self.df * self._q_fn(s1_batch).max(dim = 1).values * run_batch + rewards_batch

#             loss = loss_fn(pred, target)
#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_value_(self._q_fn.parameters(), 100)
#             optimizer.step()

#             if params['use_ddqn']:
#                 t_fn_state_dict = _t_fn.state_dict()
#                 q_fn_state_dict = self._q_fn.state_dict()
#                 for key in q_fn_state_dict:
#                     t_fn_state_dict[key] = params['tau'] * q_fn_state_dict[key] + (1 - params['tau']) * t_fn_state_dict[key]
#                 _t_fn.load_state_dict(t_fn_state_dict)

#             with open(self.log_file, 'a') as f:
#                 f.write(f'episode {episode_idx} epsilon: {eps:.5f}\n') 

#             if episode_idx % params['validate_every'] == 0:
#                 # print(f'average {(time() - t_start) / episode_idx:.2f} seconds per episode')
#                 if self.obs_type == 'discrete':
#                     self.__make_policy_discrete()
#                 rewards_avg, dist_avg = validate_policy(self, validation_runs = params['validation_runs'])
#                 print(f'episode {episode_idx:>6} rewards: {rewards_avg:>10.5f}, dist: {dist_avg:>10.5f}') 
#                 with open(self.log_file, 'a') as f:
#                     f.write(f'episode: {episode_idx} rewards: {rewards_avg:.5f}, dist: {dist_avg:.5f}\n') 
#                 self._use_policy = False



#     def choose_action(self, state: list, greedy: bool):
#         '''
#         Right now returning random action but need to add
#         your own logic
#         '''
#         if not greedy:
#             return np.random.randint(0, 5)
#         if self._use_policy:
#             return self._policy[tuple(state)]
#         with torch.no_grad():
#             return self._q_fn(torch.tensor(state).unsqueeze(0)).max(dim = 1).indices.item()
#     # TODO: increase batch size based on validation performance?
    

# if __name__ == '__main__':

#     '''
#     For part b:
#         env = get_highway_env(dist_obs_states = 5, reward_type = 'dist', obs_type='continious')
#     '''


#     args =  {
#         'env'  : None,
#         'alpha': 0.1,
#         # 'eps_args' : {
#         #     'type' : 'exponential',
#         #     'init' : 1.00,
#         #     'itr_eps_half' : 150000
#         # },

#         'eps_args' : {
#             'type' : 'linear',
#             'init' : 1.00,
#             'itr_eps_zero' : 400000,
#             'eps_min' : 0.3
#         },

#         # 'eps_args' : {
#         #     'type' : 'constant',
#         #     'init' : 0.75,
            
#         # },


#         'discount_factor' : 0.98,
#         'iterations' : 500000,
#         'validate_every' : 25000,
#         'log_folder' : './res',
#         'batch_size': 512,
#         'tau' : 0.005,
#         # 'tau' : 1.000,
#         'use_ddqn' : True,
#         'obs_type' : 'continuous',
#         # 'obs_type' : 'discrete',

#     }
#     eps_str = ''
#     for key, val in args['eps_args'].items():
#         eps_str += f'{key}_{val}_'
#     eps_str = eps_str[:-1]
#     train_arch = 'ddqn' if args['use_ddqn'] else 'dqn'
#     loss = 'mse'
#     name = 'Only_AdamW_no_w_big'
#     # name = 'ssws'

#     if args['obs_type'] == 'continuous':
#         args['state_embed_dim'] = 4
#     elif args['obs_type'] == 'discrete':
#         args['state_embed_dim'] = 16
#     else:
#         raise NotImplementedError(f'obs_type {args["obs_type"]} not implemented')
    
#     expname = f"{name}_{loss}_{args['batch_size']}_df_{args['discount_factor']}_{train_arch}_{args['obs_type']}_eps_{eps_str}_emb_{args['state_embed_dim']}"
#     args['log_folder'] = f'results-big/{expname}'

#     args['env'] = get_highway_env(dist_obs_states = 5, reward_type = 'dist', obs_type = args['obs_type'])
#     agent = BestAgent(**args)
