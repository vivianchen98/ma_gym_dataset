import pprint
from util import load, record

# argparse
import argparse
parser = argparse.ArgumentParser(description='Random Agent for ma-gym')
parser.add_argument('--env', default='PredatorPrey5x5-v0',
                    help='Name of the environment (default: %(default)s)')
parser.add_argument('--episodes', type=int, default=1,
                    help='episodes (default: %(default)s)')
args = parser.parse_args()

def compute_hist(trajs_capped, n_agents, max_horizon):
    print("Computed histogram of state-action pairs")
    hist_all = {}
    for i in range(n_agents):
        hist_t = {}
        for t in range(max_horizon):
            list = [(trajs_capped[ep][t]['action_n'][i], trajs_capped[ep][t]['obs_n'][i]) for ep in range(len(trajs_capped))]
            freq = {}
            for l in list:
                if l not in freq.keys():
                    freq[l] = 1
                else:
                    freq[l] += 1
            hist = {k: v/len(trajs_capped) for k, v in freq.items()}

            hist_t[t] = hist
        
        hist_all[i] = hist_t
    return hist_all


# import trajs_capped with k trajectories, each with a horizon of max_horizon
env, trajs, trajs_capped, max_horizon = load('recordings/'+args.env+'/data_'+str(args.episodes)+'.pickle')
print('length of trajs', max_horizon)
print('number of trajs', len(trajs_capped))

# compute histogram of state-action pairs for each player i at each timestep t
hist_all = compute_hist(trajs_capped, env.n_agents, max_horizon)
# pprint.pprint(hist_all[0])

# record processed data
record([hist_all], directory='recordings/'+args.env, label='processed_data_'+str(args.episodes))