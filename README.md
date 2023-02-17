# MA GYM INTERACTIVE DATASET
Repo with scripts to collect interative data from ma_gym environments.


## Dependencies
* gym
* ma_gym: https://github.com/koulanurag/ma-gym


## Data Generation
Collect trajectories by
```
python3 interactive_agent.py --env PredatorPrey5x5-v0 --episodes 10
```
and when you finish all episodes, you should see your trajectory stored in `data/env/interactive_data_[current_datetime].pickle`

<!-- Data for `PredatorPrey5x5-v0` is pre-generated, processed, and converted to julia JLD2 files in `ma_gym/PredatorPrey5x5-v0/julia` directory. -->
