import os
import pickle, itertools
import numpy as np

def record(data, directory="data/", label="data"):
    if not os.path.exists(directory): os.mkdir(directory)
    with open(os.path.join(directory, label+'.pickle'), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved data to `"+directory+'/'+label+'.pickle`')


def load(directory):
    with open(directory, 'rb') as handle:
        data = pickle.load(handle)
    return data


def prune_and_cap_trajs(trajs):
    # prune trajs to be all long enough
    # print('trajs:', [len(trajs[i]) for i in range(len(trajs))])
    long_enough = np.percentile([len(trajs[i]) for i in range(len(trajs))], 50)
    print("50th precentile: "+str(int(long_enough)))
    trajs_pruned = []
    for i in range(len(trajs)):
        if len(trajs[i]) >= long_enough:
            trajs_pruned += [trajs[i]]
    trajs_pruned = {k: v for k, v in enumerate(trajs_pruned)}

    # cap trajs by shortest lengths
    max_horizon = min([len(trajs_pruned[i]) for i in range(len(trajs_pruned))])
    # print("Capped trajs to horizon <=", max_horizon)
    trajs_capped = {k: v for k, v in enumerate([dict(itertools.islice(trajs_pruned[i].items(),max_horizon)) for i in range(len(trajs_pruned))])}
    # print('trajs_capped:', [len(trajs_capped[i]) for i in range(len(trajs_capped))])

    print("A total of "+str(len(trajs_capped))+" useful trajs!")

    return trajs_capped, max_horizon


def compute_hist(trajs_capped, n_players, max_horizon):
    # print("Computed histograms of joint probs, init state dist, transition.")

    # compute hist for reference joint probability dist of state-action pairs
    hist_all = {}
    for i in range(n_players):
        hist_t = {}
        for t in range(max_horizon):
            list = [(trajs_capped[ep][t]['action_n'][i], trajs_capped[ep][t]['pos_n'][i]) for ep in range(len(trajs_capped))]

            freq = {}
            for l in list:
                if l not in freq.keys():
                    freq[l] = 1
                else:
                    freq[l] += 1
            hist = {k: v/len(trajs_capped) for k, v in freq.items()}
            hist_t[t] = hist
        hist_all[i] = hist_t
    
    # compute initial state distribution
    hist_zero = {}
    for i in range(n_players):
        total_i = 0
        count_zero_i = {}
        for ep in range(len(trajs_capped)):
            s = trajs_capped[ep][0]['pos_n'][i]
            if s not in count_zero_i.keys():
                count_zero_i[s] = 1
            else:
                count_zero_i[s] += 1
            total_i += 1
        hist_zero[i] = {k: v/total_i for k,v in count_zero_i.items()}

    # compute transition probabilities
    hist_trans={}
    for i in range(n_players):
        count_sas = {}
        count_sa = {}
        for ep in range(len(trajs_capped)):
            for t in range(max_horizon-1):
                s = trajs_capped[ep][t]['pos_n'][i]
                a = trajs_capped[ep][t]['action_n'][i]
                s_next = trajs_capped[ep][t+1]['pos_n'][i]

                if (s,a) not in count_sa.keys():
                    count_sa[(s,a)] = 1
                else:
                    count_sa[(s,a)] += 1
                
                if (s,a,s_next) not in count_sas.keys():
                    count_sas[(s,a,s_next)] = 1
                else:
                    count_sas[(s,a,s_next)] += 1
        hist_trans[i] = {(s,a,s_next): v / count_sa[(s,a)] for (s,a,s_next),v in count_sas.items()}

    return hist_all, hist_zero, hist_trans