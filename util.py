import os
import pickle, itertools
import numpy as np

def record(data, directory="recordings/", label="data"):
    if not os.path.exists("recordings"): os.mkdir("recordings")
    if not os.path.exists(directory): os.mkdir(directory)
    with open(os.path.join(directory, label+'.pickle'), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(directory):
    with open(directory, 'rb') as handle:
        data = pickle.load(handle)
    return data


def prune_and_cap_trajs(trajs):
    # prune trajs to be all long enough
    long_enough = np.percentile([len(trajs[i]) for i in range(len(trajs))], 10)
    print("traj of length "+str(long_enough)+" is long enough")
    trajs_pruned = []
    for i in range(len(trajs)):
        if len(trajs[i]) >= long_enough:
            trajs_pruned += [trajs[i]]
    trajs_pruned = {k: v for k, v in enumerate(trajs_pruned)}

    # cap trajs by shortest lengths
    # print("lengths of pruned trajs:", [len(trajs_pruned[i]) for i in range(len(trajs_pruned))])
    max_horizon = min([len(trajs_pruned[i]) for i in range(len(trajs_pruned))])
    print("Capped trajs to horizon <=", max_horizon)
    trajs_capped = {k: v for k, v in enumerate([dict(itertools.islice(trajs_pruned[i].items(),max_horizon)) for i in range(len(trajs_pruned))])}
    print("A total of "+str(len(trajs_capped))+" useful trajs!")

    return trajs_capped, max_horizon
