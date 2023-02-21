import argparse
import collections
import os
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from util import record

def compute_prey_action(old_pos, new_pos):
    movement = (new_pos[0] - old_pos[0], new_pos[1] - old_pos[1]) 
    if movement == (0,0):
        action = 4 # noop
    elif movement == (0, -1):
        action = 1 # left
    elif movement == (0, 1):
        action = 3 # right
    elif movement == (-1, 0):
        action = 2 # up
    elif movement == (1, 0):
        action = 0 # down
    return action


def collect(env, collect_episodes, q, type):
    trajs = []
    for ep_i in range(collect_episodes):
        traj = []
        done_n = [False for _ in range(env.n_agents)]
        ep_reward = 0
        obs_n = env.reset()
        # env.render()

        with torch.no_grad():
            if type == 'vdn' or type == 'qmix': 
                hidden = q.init_hidden()

            while not all(done_n):
                old_prey_pos = env.prey_pos

                if type == 'idqn':
                    action_n = q.sample_action(torch.Tensor(obs_n).unsqueeze(0), epsilon=0)
                elif type == 'vdn' or 'qmix':
                    action_n, hidden = q.sample_action(torch.Tensor(obs_n).unsqueeze(0), hidden, epsilon=0)

                obs_n, reward_n, done_n, info = env.step(action_n[0].data.cpu().numpy().tolist())
                traj += [{"pos_n": [tuple(env.agent_pos[i]) for i in range(env.n_agents)]+ [tuple(env.prey_pos[i]) for i in range(env.n_preys)], 
                      "action_n": [int(_) for _ in action_n.tolist()[0]] + [compute_prey_action(old_prey_pos[i], env.prey_pos[i]) for i in range(env.n_preys)]}]
                ep_reward += sum(reward_n)
                # env.render()

        trajs += [{k: v for k, v in enumerate(traj)}]
        print('Episode #{} Reward: {}'.format(ep_i, ep_reward))

    return trajs

# python3 collect_PredatorPrey_algo.py --env-name PredatorPrey5x5-v1 --type [vdn/qmix/idqn/maddpg] --collect-episodes 100
if __name__ == '__main__':
    # Lets gather arguments
    parser = argparse.ArgumentParser(description='Collect trajs for ma-gym')
    parser.add_argument('--env-name', required=False, default='ma_gym:PredatorPrey5x5-v1')
    parser.add_argument('--type', required=False, default='vdn')
    parser.add_argument('--max-episodes', type=int, default=15000, required=False)
    parser.add_argument('--collect-episodes', type=int, default=100, required=False)

    # Process arguments
    args = parser.parse_args()

    # create env
    env = gym.make(args.env_name)

    # import approriate models
    if args.type == 'vdn':
        from scripts.vdn import QNet
        q = QNet(env.observation_space, env.action_space, True)
    elif args.type == 'qmix':
        from scripts.qmix import QNet
        q = QNet(env.observation_space, env.action_space, True)
    elif args.type == 'idqn':
        from scripts.idqn import QNet
        q = QNet(env.observation_space, env.action_space)
    else:
        print("Type unsupported!")

    # load model q
    model_dir = 'models/'+args.env_name+'_'+args.type+'_'+str(args.max_episodes)+'.pth'
    q.load_state_dict(torch.load(model_dir))
    q.eval()

    # simulate policy
    trajs = collect(env, args.env_name, args.collect_episodes, q, args.type)

    # close env
    env.close()

    # record raw data
    if not os.path.exists('raw_data'): os.mkdir('raw_data')
    record([env, trajs], directory='raw_data/'+args.env_name, label=args.type+'_'+str(args.collect_episodes))
