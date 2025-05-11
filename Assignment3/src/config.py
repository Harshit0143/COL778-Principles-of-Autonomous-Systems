
'''
This file contains the configs used for Model creation and training. You need to give your best hyperparameters and the configs you used to get the best results for 
every environment and experiment.  These configs will be automatically loaded and used to create and train your model in our servers.
'''
#You can add extra keys or modify to the values of the existing keys in bottom level of the dictionary.
#DO NOT CHANGE THE STRUCTURE OF THE DICTIONARY. 

configs = {
    
    'Hopper-v4': {
            #You can add or change the keys here
              "hyperparameters": {
                'hidden_size': 128,
                'n_layers': 3,
                'batch_size': 512,
                'learning_rate': 0.001,
                'min_timesteps_per_batch': 1000,
                'beta': 0.10,
                'savename': 'Hopper-v4.pth',
                'save_every': 1,
            },
            "num_iteration": 10000,
    },
    
    
    'Ant-v4': {
               "hyperparameters": {
                'hidden_size': 128,
                'n_layers': 5,
                'batch_size': 512,
                'learning_rate': 0.001,
                'min_timesteps_per_batch': 1000,
                'beta': 0.10,
                'savename': 'Ant-v4.pth',
                'save_every': 1,
            },
            "num_iteration": 10000,
    }
}