#!/bin/bash

for env in PredatorPrey5x5-v0 PredatorPrey7x7-v0
do
    for episodes in 1 50 100
    do
        echo "$env, episodes=$episodes"
        python3 random_agent.py --env $env --episodes $episodes
    done
done