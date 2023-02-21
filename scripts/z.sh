#!/bin/bash

# python3 qmix.py --env-name ma_gym:PredatorPrey5x5-v0 --max-episodes 15000
# python3 vdn.py --env-name ma_gym:PredatorPrey5x5-v0 --max-episodes 15000
# python3 idqn.py --env-name ma_gym:PredatorPrey5x5-v0 --max-episodes 15000

# python3 qmix.py --env-name ma_gym:PredatorPrey5x5-v1 --max-episodes 15000
# python3 vdn.py --env-name ma_gym:PredatorPrey5x5-v1 --max-episodes 15000
# python3 idqn.py --env-name ma_gym:PredatorPrey5x5-v1 --max-episodes 15000

python3 maddpg.py --env-name ma_gym:PredatorPrey5x5-v0 --max-episodes 15000
python3 maddpg.py --env-name ma_gym:PredatorPrey5x5-v1 --max-episodes 15000