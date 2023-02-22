import argparse, os, random
from datetime import datetime
import gym
import torch
import torch.nn as nn
# from ma_gym.wrappers import Monitor
from util import record
# from pprint import pprint

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


def collect_random(env, collect_episodes):
    trajs = []
    for ep_i in range(collect_episodes):
        traj = []
        done_n = [False for _ in range(env.n_agents)]
        ep_reward = 0
        env.seed(ep_i)
        obs_n = env.reset()
        # env.render()

        while not all(done_n):
            old_prey_pos = env.prey_pos
            action_n = env.action_space.sample() # random action_n
            obs_n, reward_n, done_n, info = env.step(action_n)
            traj += [{"pos_n": [tuple(env.agent_pos[i]) for i in range(env.n_agents)]+ [tuple(env.prey_pos[i]) for i in range(env.n_preys)], 
                      "action_n": action_n + [compute_prey_action(old_prey_pos[i], env.prey_pos[i]) for i in range(env.n_preys)]}]

            ep_reward += sum(reward_n)
            # env.render()

        print('Episode #{} Reward: {}'.format(ep_i, ep_reward))
        trajs += [{k: v for k, v in enumerate(traj)}]

    return trajs


def collect_interactive(env, collect_episodes):
    trajs = []
    for ep_i in range(collect_episodes):
        traj = []
        done_n = [False for _ in range(env.n_agents)]
        ep_reward = 0
        env.seed(ep_i)
        obs_n = env.reset()
        env.render()

        while not all(done_n):
            old_prey_pos = env.prey_pos

            # interactive action_n
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

            obs_n, reward_n, done_n, info = env.step(action_n)

            traj += [{"pos_n": [tuple(env.agent_pos[i]) for i in range(env.n_agents)]+ [tuple(env.prey_pos[i]) for i in range(env.n_preys)], 
                      "action_n": action_n + [compute_prey_action(old_prey_pos[i], env.prey_pos[i]) for i in range(env.n_preys)]}]

            ep_reward += sum(reward_n)
            env.render()

        print('Episode #{} Reward: {}'.format(ep_i, ep_reward))
        trajs += [{k: v for k, v in enumerate(traj)}]

    return trajs


def collect_algo(env, collect_episodes, model, type):
    trajs = []
    for ep_i in range(collect_episodes):
        traj = []
        done_n = [False for _ in range(env.n_agents)]
        ep_reward = 0
        env.seed(random.randrange(1000))
        obs_n = env.reset()
        # env.render()

        with torch.no_grad():
            if type == 'vdn' or type == 'qmix': 
                hidden = model.init_hidden()

            while not all(done_n):
                old_prey_pos = env.prey_pos

                if type == 'idqn':
                    action_n = model.sample_action(torch.Tensor(obs_n).unsqueeze(0), epsilon=0)
                elif type == 'vdn' or type == 'qmix':
                    action_n, hidden = model.sample_action(torch.Tensor(obs_n).unsqueeze(0), hidden, epsilon=0)
                elif type == 'maddpg':
                    action_logits = model(torch.Tensor(obs_n).unsqueeze(0))
                    action_n = action_logits.argmax(dim=2)

                obs_n, reward_n, done_n, info = env.step(action_n[0].data.cpu().numpy().tolist())
                traj += [{"pos_n": [tuple(env.agent_pos[i]) for i in range(env.n_agents)]+ [tuple(env.prey_pos[i]) for i in range(env.n_preys)], 
                      "action_n": [int(_) for _ in action_n.tolist()[0]] + [compute_prey_action(old_prey_pos[i], env.prey_pos[i]) for i in range(env.n_preys)]}]
                ep_reward += sum(reward_n)
                # env.render()

        trajs += [{k: v for k, v in enumerate(traj)}]
        print('Episode #{} Reward: {}'.format(ep_i, ep_reward))

    # pprint({k:v for k,v in enumerate(trajs)})
    return trajs


# python3 collect_PredatorPrey.py --env ma_gym:PredatorPrey5x5-v1 --type [interactive/random/vdn/qmix/idqn/maddpg] --collect-episodes 10
if __name__ == '__main__':
    # Lets gather arguments
    parser = argparse.ArgumentParser(description='Collect trajs')
    parser.add_argument('--env-name', default='ma_gym:PredatorPrey5x5-v1',
                        help='Name of the environment (default: %(default)s)')
    parser.add_argument('--type', default='random',
                        help='[random/interactive/vdn/qmix/idqn/maddpg] (default: %(default)s)')    
    parser.add_argument('--max-episodes', type=int, default=15000, required=False, help='(default: %(default)s)')
    parser.add_argument('--collect-episodes', type=int, default=100, required=False,
                        help='Number of episode to collect (default: %(default)s)')   

    # Process arguments
    args = parser.parse_args()

    # create env
    env = gym.make(args.env_name)
    # env = Monitor(env, directory=args.env + '/monitor', force=True)

    if args.type == 'interactive':
        trajs = collect_interactive(env, args.collect_episodes)
    elif args.type == 'random':
        trajs = collect_random(env, args.collect_episodes)
    else:
        # import approriate models
        if args.type == 'vdn':
            from scripts.vdn import QNet
            model = QNet(env.observation_space, env.action_space, True)
        elif args.type == 'qmix':
            from scripts.qmix import QNet
            model = QNet(env.observation_space, env.action_space, True)
        elif args.type == 'idqn':
            from scripts.idqn import QNet
            model = QNet(env.observation_space, env.action_space)
        elif args.type == 'maddpg':
            from scripts.maddpg import MuNet
            model = MuNet(env.observation_space, env.action_space)
        else:
            print("Type unsupported!")
        
        # load model
        model_dir = 'models/'+args.env_name+'_'+args.type+'_'+str(args.max_episodes)+'.pth'
        model.load_state_dict(torch.load(model_dir))
        model.eval()

        # simulate policy
        trajs = collect_algo(env, args.collect_episodes, model, args.type)
    
    # close env
    env.close()

    # record raw data
    if not os.path.exists('raw_data'): os.mkdir('raw_data')
    record([env, trajs], directory='raw_data/'+args.env_name, label=args.type+'_'+str(args.collect_episodes)+'_'+str(datetime.now()))
