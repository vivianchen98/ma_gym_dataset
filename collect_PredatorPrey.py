import argparse, os
import gym
from util import record
# from ma_gym.wrappers import Monitor
from datetime import datetime

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


# python3 collect_PredatorPrey.py --env PredatorPrey5x5-v0 --type [random/interactive] --episodes 10

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random Agent for ma-gym')
    parser.add_argument('--env', default='PredatorPrey5x5-v0',
                        help='Name of the environment (default: %(default)s)')
    parser.add_argument('--episodes', type=int, default=1,
                        help='episodes (default: %(default)s)')
    parser.add_argument('--type', default='interactive',
                        help='type (default: %(default)s)')                        
    args = parser.parse_args()

    
    env = gym.make('ma_gym:' + args.env)
    # env = Monitor(env, directory=args.env + '/monitor', force=True)

    if args.type == 'interactive':
        print(list(enumerate(env.get_action_meanings(1))))


    trajs = []
    for ep_i in range(args.episodes):
        traj = []
        done_n = [False for _ in range(env.n_agents)]
        ep_reward = 0
        env.seed(ep_i)
        obs_n = env.reset()
        # env.render()

        while not all(done_n):
            old_prey_pos = env.prey_pos

            if args.type == 'random':
                action_n = env.action_space.sample()
            elif args.type == 'interactive':
                action_n = [int(_) for _ in input('Action:')]

            obs_n, reward_n, done_n, info = env.step(action_n)

            traj += [{"pos_n": [tuple(env.agent_pos[i]) for i in range(env.n_agents)]+ [tuple(env.prey_pos[i]) for i in range(env.n_preys)], 
                      "action_n": action_n + [compute_prey_action(old_prey_pos[i], env.prey_pos[i]) for i in range(env.n_preys)]}]

            ep_reward += sum(reward_n)
            # env.render()

        print('Episode #{} Reward: {}'.format(ep_i, ep_reward))
        traj = {k: v for k, v in enumerate(traj)}
        trajs += [traj]

    env.close()

    # record raw data
    if not os.path.exists('raw_data'): os.mkdir('raw_data')
    record([env, trajs], directory='raw_data/'+args.env, label=args.type+'_'+str(datetime.now()))