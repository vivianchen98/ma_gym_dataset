import os, argparse
from util import load, prune_and_cap_trajs, compute_hist, record


# python3 process.py --type vdn
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process trajs for a specific type')
    parser.add_argument('--type', default='random',
                        help='[random/interactive/vdn/qmix/idqn/maddpg] (default: %(default)s)')                        
    args = parser.parse_args()

    envs = os.listdir("raw_data")
    for e in envs:
        # concatenate all trajs in the same env
        trajs_all = []
        print(e)
        for file in os.listdir("raw_data/"+e):
            if file.startswith(args.type) and file.endswith(".pickle"):
                print('\t'+file)
                env, trajs = load('raw_data/'+e+'/'+file)
                trajs_all += trajs
        # print()

        if len(trajs_all) > 0:
            # processing
            trajs_all = {k: v for k, v in enumerate(trajs_all)}
            trajs_capped, max_horizon = prune_and_cap_trajs(trajs_all)
            hist_all, hist_zero, hist_trans = compute_hist(trajs_capped, env.n_agents+env.n_preys, max_horizon)

            # record processed data
            if not os.path.exists('processed_data'): os.mkdir('processed_data')
            record([env, trajs_all, trajs_capped, max_horizon, hist_all, hist_zero, hist_trans], directory='processed_data/'+e, label=args.type+'_all')
            print()
        else:
            print("No data needed to process!")
            print()