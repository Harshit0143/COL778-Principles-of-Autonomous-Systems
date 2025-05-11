
## Train Tabular Q-Learning
```
python tabular_q.py --iterations 100000 --output_folder results_tab --visualize_runs 10 --validate_every 1000 --validation_runs 1000
```
## Train DQN Agent
```
python dqn.py --iterations 200000 --output_folder results_cts --visualize_runs 10 --validate_every 1000 --validation_runs 1000 --obs_type discrete
python dqn.py --iterations 200000 --output_folder results_cts --visualize_runs 10 --validate_every 1000 --validation_runs 1000 --obs_type continuous
```

## Train `BestAgent`
```
python part_3_eval.py --iterations 500000 --input_file input.txt --output_file output.txt
```
## Run all experimnts
```
python experiments.py
```

## Run inference [validation. speed and lane visulizations]
```
python inference.py
```
