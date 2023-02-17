import os
from util import load, prune_and_cap_trajs, compute_hist, record


if __name__ == '__main__':
    envs = os.listdir("data")
    for e in envs:
        trajs_all = []
        print(e)
        for file in os.listdir("data/"+e):
            if file.endswith("all.pickle"):
                os.remove('data/'+e+'/'+file)
            elif file.endswith(".pickle"):
                print('\t'+file)
                env, trajs = load('data/'+e+'/'+file)
                trajs_all += trajs
        trajs_all = {k: v for k, v in enumerate(trajs_all)}
        trajs_capped, max_horizon = prune_and_cap_trajs(trajs_all)
        hist_all, hist_zero, hist_trans = compute_hist(trajs_capped, env.n_agents, max_horizon)
        record([env, trajs_all, trajs_capped, max_horizon, hist_all, hist_zero, hist_trans], directory='data/'+e, label='interactive_data_all')
        print("Saved trajs for " + e +"\n")


    # env, trajs = load(args.env+'/random_data_'+args.episodes+'.pickle')
    # trajs_capped, max_horizon = prune_and_cap_trajs(trajs)
    # hist_all, hist_zero, hist_trans = compute_hist(trajs_capped, env.n_agents, max_horizon)
