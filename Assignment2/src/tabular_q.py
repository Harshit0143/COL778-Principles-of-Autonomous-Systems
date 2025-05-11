
import os
import pickle
import numpy as np 
from itertools import product
from env import HighwayEnv, get_highway_env
from utils import show_params, get_epsilon, validate_policy, visualize_policy, parse_args

class TabularQAgent:
    
    def __init__(self, 
                    env: HighwayEnv, 
                    eps_args: dict = {'type': 'constant', 'init': 0.75},
                    alpha: float = 0.1, 
                    discount_factor: float = 0.9,
                    iterations: int = 10000, 
                    log_folder: str = 'results/experiments/tabular_q',
                    validation_runs: int = 1000,
                    validate_every: int = 1000,
                    pretrained: str = None
                ):


        self.env = env
        if pretrained is not None:
            metadata = pickle.load(open(pretrained, 'rb'))
            self.df = metadata['df']
            self._q_table = metadata['_q_table']
            self.__make_policy()
            return


        self.df = discount_factor
        self.log_file = os.path.join(log_folder, 'train_logs.txt')
        os.makedirs(log_folder, exist_ok = True)
        params = {
            'df': self.df,
            'iterations': iterations,
            'alpha': alpha,
            'eps_args': eps_args,
            'validation_runs': validation_runs,
            'validate_every': validate_every

        }
        show_params(params, self.log_file)
        self._q_table = self.__get_empty_q_table()
        self._eval = False
        self.__train_policy(params)


        savepath = os.path.join(log_folder, 'q_tabular_agent.pkl')
        with open(savepath, 'wb') as f:
            pickle.dump({
                'df': self.df,
                '_q_table': self._q_table,
            }, f)
        with open(self.log_file, 'a') as f:
            f.write(f'Model saved to {savepath} after {params["iterations"]} iterations\n')


    def __get_empty_q_table(self):
        keys = product({1, 2, 3, 4}, range(4), range(5), range(5), range(5), range(5))
        return {key: [0, 0, 0, 0, 0] for key in keys}
    
    def __make_policy(self):
        self._eval = True
        self._policy = {key: np.argmax(val) for key, val in self._q_table.items()}
    
    def __train_policy(self, params):
        '''
        Learns the policy
        '''
        #TO DO: You can add you code here
        print(f'Learning policy for {params["iterations"]} iterations...' )
        for episode_idx in range(1, params['iterations'] + 1):
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
                f.write(f'episode {episode_idx} epsilon: {eps:.5f}\n') 

            if episode_idx % params['validate_every'] == 0:
                self.__make_policy()
                rewards_avg, dist_avg = validate_policy(self, validation_runs = params['validation_runs'])
                print(f'episode {episode_idx} rewards: {rewards_avg:.5f}, dist: {dist_avg:.5f}') 
                with open(self.log_file, 'a') as f:
                    f.write(f'episode {episode_idx} rewards: {rewards_avg:.5f}, dist: {dist_avg:.5f}\n') 
                self._eval = False
                


    def choose_action(self, state, greedy):
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
    args = parse_args()
    env = get_highway_env(dist_obs_states = 5, reward_type = 'dist')
    assert args.obs_type == 'discrete', f'obs_type {args.obs_type} not supported for tabular Q-learning'
    qagent = TabularQAgent(env = env, 
                           log_folder = args.output_folder,
                           iterations = args.iterations,
                           validate_every = args.validate_every,
                           validation_runs = args.validation_runs)
    
    visualize_policy(qagent, args.visualize_runs, f'{args.output_folder}/gifs')
