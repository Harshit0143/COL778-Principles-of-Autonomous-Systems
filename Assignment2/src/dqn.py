
SEED = 42
import os
import torch
import random
import pickle
import argparse
import numpy as np 
from torch import nn
torch.manual_seed(SEED)
random.seed(SEED)  
from itertools import product
from collections import OrderedDict, deque
from env import HighwayEnv, get_highway_env
import matplotlib.pyplot as plt
from copy import deepcopy
import re
from PIL import Image
from utils import show_params, get_epsilon, validate_policy, visualize_policy, parse_args


class QFunction(nn.Module):
    def __init__(self, state_embed_dim, hidden_dim, mode):
        super(QFunction, self).__init__()
        num_speed, num_lane, num_dist = 4, 4, 5
        action_dim = 5
        if mode == 'discrete':
            state_dim = num_speed * num_lane * (num_dist ** 4)
            self.weights = torch.tensor([
                num_lane * (num_dist ** 4),
                num_dist ** 4,
                num_dist ** 3,
                num_dist ** 2,
                num_dist ** 1,
                1
                ], dtype = torch.int64)

            self.encode_fn = self.encode_discrete
            ff_input_dim = state_embed_dim

        elif mode == 'continuous':
            state_dim = num_speed * num_lane
            self.weights = torch.tensor([
                num_lane,
                1
            ], dtype = torch.int64)
            self.encode_fn = self.encode_continious
            ff_input_dim = state_embed_dim + 4
        
        else:
            raise(NotImplementedError(f'obs_type {mode} not implemented'))
        
        self.encode_layer = nn.Embedding(num_embeddings = state_dim,  embedding_dim = state_embed_dim)

     
        self.layers = nn.Sequential(OrderedDict([
            ('input  ', nn.Linear(ff_input_dim, hidden_dim)),
            ('relu1  ', nn.ReLU()),
            ('hidden1', nn.Linear(hidden_dim, hidden_dim)),
            ('relu2  ', nn.ReLU()),
            ('hidden2', nn.Linear(hidden_dim, hidden_dim)),
            ('relu3  ', nn.ReLU()),
            ('output ', nn.Linear(hidden_dim, action_dim))
        ]))

    def encode_continious(self, state):
        state[: , 0] -= 1
        return torch.cat((self.encode_layer(torch.matmul(state[:, : 2].long(), self.weights)), state[:, 2 : ]), dim = 1)

    def encode_discrete(self, state):
        state[:, 0] -= 1
        return self.encode_layer(torch.matmul(state, self.weights))



        
    def forward(self, state):
        state = self.encode_fn(state)
        return self.layers(state)
    



class DQNAgent:
    def __init__(self, 
                    env: HighwayEnv, 
                    eps_args: dict = {'type': 'constant', 'init': 0.75},
                    alpha: float = 0.1,  
                    discount_factor: float = 0.9,
                    iterations: int = 10000, 
                    log_folder: str = 'results/experiments/dqn',
                    obs_type: str = 'discrete',
                    lr: float = 0.0001,
                    tau: float = 0.005,
                    batch_size: int = 256,
                    buffer_size: int = 10000,
                    hidden_dim: int = 32, 
                    validation_runs: int = 1000,
                    validate_every: int = 10000,
                    state_embed_dim: int = 16,
                    pretrained: str = None
    ):

        keys = product({1, 2, 3, 4}, range(4), range(5), range(5), range(5), range(5))
        self.states = [key for key in keys]

        self.env = env 
        if pretrained is not None:
            metadata = pickle.load(open(pretrained, 'rb'))
            self.df = metadata['df']
            self._q_fn = QFunction(state_embed_dim = metadata['state_embed_dim'],
                                hidden_dim = metadata['hidden_dim'], 
                                mode = metadata['obs_type'])
            self._q_fn.load_state_dict(torch.load(metadata['state_dict_path']))

            if metadata['obs_type'] == 'discrete':
                self.__make_policy_discrete()
            else:
                self._use_policy = False
            return


        self.df = discount_factor
        
        self.log_file = os.path.join(log_folder, f'train_logs.txt')
        os.makedirs(log_folder, exist_ok = True)
        self.obs_type = obs_type

        params = {
            'df': self.df,
            'obs_type': self.obs_type,
            'iterations': iterations,
            'alpha': alpha,
            'eps_args': eps_args,
            'validation_runs': validation_runs,
            'validate_every': validate_every,
            'state_embed_dim': state_embed_dim,
            'hidden_dim': hidden_dim,
            'lr': lr,
            'batch_size': batch_size,
            'buffer_size': buffer_size,
            'tau': tau,
        }
        show_params(params, self.log_file)


        
        

        self._q_fn = QFunction(state_embed_dim = state_embed_dim, hidden_dim = hidden_dim, mode = obs_type) 

        
        # encoding = self._q_fn.encode_fn(torch.tensor(self.states))
        # assert (encoding[1 : ] - encoding[: -1]).min() > 0
                
        self._use_policy = False
        self.__train_policy(params)
        
        if self.obs_type == 'discrete':
            self.__make_policy_discrete()
        
        savepath = os.path.join(log_folder, 'dqn_agent')
        state_dict_savepath = f'{savepath}.pth'
        metadata = {
            'df': self.df,
            'obs_type': self.obs_type,
            'state_embed_dim': state_embed_dim,
            'hidden_dim': hidden_dim,
            'state_dict_path': state_dict_savepath
        }
        torch.save(self._q_fn.state_dict(), state_dict_savepath)
        with open(f'{savepath}.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        with open(self.log_file, 'a') as f:
            f.write(f'metadatal saved to {savepath}.pkl after {params["iterations"]} iterations\n')


        


    


    def __make_policy_discrete(self):
        self._use_policy = True
        with torch.no_grad():
            output = self._q_fn(torch.tensor(self.states)).max(dim = 1).indices
        self._policy = {}
        for idx in range(len(self.states)):
            self._policy[tuple(self.states[idx])] = output[idx].item()




    
    def __train_policy(self, params):
        '''
        Learns the policy
        '''
        print(f'Learning policy for {params["iterations"]} iterations...' )
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(self._q_fn.parameters(), lr = params['lr'])
        buffer = deque(maxlen = params['buffer_size'])
        for episode_idx in range(1, params['iterations'] + 1):
            eps = get_epsilon(params['eps_args'], episode_idx)
            state = self.env.reset(episode_idx)
            episode_finished = False
            
            while not episode_finished:
                action = self.choose_action(state, greedy = np.random.rand() > eps)
                s_new, reward, episode_finished, _ = self.env.step(action)
                buffer.append((state, action, s_new, reward, int(not episode_finished)))
                state = s_new

            batch = random.sample(buffer, min(len(buffer), params['batch_size']))
            s0_batch, action_batch, s1_batch, rewards_batch, run_batch = zip(*batch)
            s0_batch, action_batch, s1_batch, rewards_batch, run_batch = torch.tensor(s0_batch), torch.tensor(action_batch), torch.tensor(s1_batch), torch.tensor(rewards_batch), torch.tensor(run_batch)
            
            pred = self._q_fn(s0_batch).gather(dim = 1, index = action_batch.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                target = self.df * self._q_fn(s1_batch).max(dim = 1).values * run_batch + rewards_batch

                    

            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    

            with open(self.log_file, 'a') as f:
                f.write(f'episode {episode_idx} epsilon: {eps:.5f}\n') 

            if episode_idx % params['validate_every'] == 0:
                if self.obs_type == 'discrete':
                    self.__make_policy_discrete()
                rewards_avg, dist_avg = validate_policy(self, validation_runs = params['validation_runs'])
                print(f'episode {episode_idx} rewards: {rewards_avg:.5f}, dist: {dist_avg:.5f}') 
                with open(self.log_file, 'a') as f:
                    f.write(f'episode {episode_idx} rewards: {rewards_avg:.5f}, dist: {dist_avg:.5f}\n') 
                self._use_policy = False

    def choose_action(self, state: list, greedy: bool):
        '''
        Right now returning random action but need to add
        your own logic
        '''
        if not greedy:
            return np.random.randint(0, 5)
        if self._use_policy:
            return self._policy[tuple(state)]
        with torch.no_grad():
            return self._q_fn(torch.tensor(state).unsqueeze(0)).max(dim = 1).indices.item()


if __name__ == '__main__':
    
    args = parse_args()
    env = get_highway_env(dist_obs_states = 5, reward_type = 'dist', obs_type = args.obs_type)
    state_embed_dim = 16 if args.obs_type == 'discrete' else 4
    qagent = DQNAgent(
                env = env,
                log_folder = args.output_folder,
                obs_type = args.obs_type,
                state_embed_dim = state_embed_dim,
                iterations = args.iterations,
                validate_every = args.validate_every,
                validation_runs = args.validation_runs,
            )
    visualize_policy(qagent, args.visualize_runs, f'{args.output_folder}/gifs')
