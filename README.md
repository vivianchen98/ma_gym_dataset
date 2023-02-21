# MA GYM DATASET
Dataset and Scripts for various types of trajectories from multi-agent gym environments, processed data can be used to reconstruct approximated policy, initial state distribution, and transition functions for Markov Games.

- [MA GYM DATASET](#ma-gym-dataset)
  - [1. Installation](#1-installation)
  - [2. Trajectory Generation](#2-trajectory-generation)
    - [2.1. Random actions](#21-random-actions)
    - [2.2. Interactive actions](#22-interactive-actions)
    - [2.3. Sample from marl models](#23-sample-from-marl-models)
  - [3. Processing trajectories](#3-processing-trajectories)
  - [4. Summary of processed data](#4-summary-of-processed-data)
  - [5. Acknowledgement](#5-acknowledgement)


## 1. Installation
`pip install ma-gym>=0.0.7 torch>=1.8 wandb`


## 2. Trajectory Generation
Collect trajectories by
```shell
ma_gym_dataset$ python3 collect_PredatorPrey.py --env PredatorPrey5x5-v1 --type [interactive/random/vdn/qmix/idqn/maddpg] --collect-episodes 10
```
and when you finish all episodes, you should see your trajectory stored in `raw_data/[env]/[type]_[collect_episodes]_[current_datetime].pickle`

### 2.1. Random actions
```python
action_n = env.action_space.sample()
```

### 2.2. Interactive actions
```python
action_n = []
            while True:
                input_n = [int(_) for _ in input('Action:')]
                if len(input_n) != 2:
                    print("Input not of size 2! Try again.")
                elif input_n[0] not in range(env.action_space[0].n) or input_n[1] not in range(env.action_space[1].n):
                    print("Please select from: ↓(0), ←(1), ↑(2), →(3), NOOP(4), try again.")
                else:
                    action_n = input_n
                    break
```

### 2.3. Sample from marl models
There are four marl algorithms used here: idqn, qmix, vdn, maddpg.
Each of them are trained for 15000 episodes on environments `PredatorPrey5x5-v0` and `PredatorPrey5x5-v1`, with models stored in folder `models/`

```python
if type == 'idqn':
    action_n = model.sample_action(torch.Tensor(obs_n).unsqueeze(0), epsilon=0)
elif type == 'vdn' or type == 'qmix':
    action_n, hidden = model.sample_action(torch.Tensor(obs_n).unsqueeze(0), hidden, epsilon=0)
elif type == 'maddpg':
    action_logits = model(torch.Tensor(obs_n).unsqueeze(0))
    action_n = action_logits.argmax(dim=2)
```

If you want to run these algorithms on new environments, please `cd scripts/` and call algorithms there, e.g.,
```shell
ma_gym_dataset/scripts$ python3 --env-name ma_gym:PredatorPrey5x5-v1 --seed 1 --max-episodes 15000
```
`wandb` is on by default, you can disable it by editing `USE_WANDB = False` in the corresponding algorithm files. The resulting models will be saved in directory `models/`.

## 3. Processing trajectories
In `util.py`, functions are defined to prune trajectories not long enough (which is current set to be the 50th percentile of lengths of trajs), and cap the remaining trajs longer than that, thus resulting in a list of trajectories with the same length. There is also a function to compute the histogram of probability distributions of state-action pairs in `hist_all`, the initial states in `hist_zero`, and state transition in `hist_trans`. All of them are in the format of dictionaries, and can be used to construct approximated policy, initial state distribution, and transition functions.

You can call processing for a particular type, e.g.,
```shell
ma_gym_dataset$ python3 process.py --type vdn
```
and it will concatenate all trajectories of the requested type in the same environment, and store them in a newly created folder called `processed_data`. You should see something like the following in terminal:
```
ma_gym:PredatorPrey5x5-v1
        vdn_200_2023-02-21 10:42:57.918056.pickle
        vdn_10_2023-02-21 10:43:17.489392.pickle
        vdn_500_2023-02-21 10:40:57.532733.pickle
        vdn_25_2023-02-21 10:43:10.697787.pickle
        vdn_5_2023-02-21 10:43:23.185119.pickle
        vdn_200_2023-02-21 10:42:52.069788.pickle
        vdn_3_2023-02-21 10:43:29.404779.pickle
50th precentile: 6
A total of 500 useful trajs!
Saved data to `processed_data/ma_gym:PredatorPrey5x5-v1/vdn_all.pickle`

PredatorPrey5x5-v0
No data needed to process!
```

## 4. Summary of processed data
| Env | Type | Number of useful trajs |
| --- | --- | --- |
| ma_gym:PredatorPrey5x5-v0 | random | 500 |
| ma_gym:PredatorPrey5x5-v0 | interactive | 100 |
| ma_gym:PredatorPrey5x5-v1 | vdn | 500 |

Only trajectories from `vdn` are collected for `PredatorPrey5x5-v1` because it performs the best among all marl algorithms in the environment. Check out [this wandb report]() to learn more.


## 5. Acknowledgement
The environments used are from [ma-gym](https://github.com/koulanurag/ma-gym), and the algorithms code in `scripts/` are from [minimal-marl](https://github.com/koulanurag/minimal-marl). 