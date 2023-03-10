import gym
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ma_gym.wrappers import Monitor
import os

USE_WANDB = True  # if enabled, logs data on wandb server


class ReplayBuffer:
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append((np.ones(len(done)) - done).tolist())

        return torch.tensor(s_lst, dtype=torch.float), \
               torch.tensor(a_lst, dtype=torch.float), \
               torch.tensor(r_lst, dtype=torch.float), \
               torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst, dtype=torch.float)

    def size(self):
        return len(self.buffer)


class MuNet(nn.Module):
    def __init__(self, observation_space, action_space):
        super(MuNet, self).__init__()
        self.num_agents = len(observation_space)
        self.action_space = action_space
        for agent_i in range(self.num_agents):
            n_obs = observation_space[agent_i].shape[0]
            num_action = action_space[agent_i].n
            setattr(self, 'agent_{}'.format(agent_i), nn.Sequential(nn.Linear(n_obs, 128),
                                                                    nn.ReLU(),
                                                                    nn.Linear(128, 64),
                                                                    nn.ReLU(),
                                                                    nn.Linear(64, num_action)))

    def forward(self, obs):
        action_logits = [torch.empty(1, _.n) for _ in self.action_space]
        for agent_i in range(self.num_agents):
            x = getattr(self, 'agent_{}'.format(agent_i))(obs[:, agent_i, :]).unsqueeze(1)
            action_logits[agent_i] = x

        return torch.cat(action_logits, dim=1)


class QNet(nn.Module):
    def __init__(self, observation_space, action_space):
        super(QNet, self).__init__()
        self.num_agents = len(observation_space)
        total_action = sum([_.n for _ in action_space])
        total_obs = sum([_.shape[0] for _ in observation_space])
        for agent_i in range(self.num_agents):
            setattr(self, 'agent_{}'.format(agent_i), nn.Sequential(nn.Linear(total_obs + total_action, 128),
                                                                    nn.ReLU(),
                                                                    nn.Linear(128, 64),
                                                                    nn.ReLU(),
                                                                    nn.Linear(64, 1)))

    def forward(self, obs, action):
        q_values = [torch.empty(obs.shape[0], )] * self.num_agents
        x = torch.cat((obs.view(obs.shape[0], obs.shape[1] * obs.shape[2]),
                       action.view(action.shape[0], action.shape[1] * action.shape[2])), dim=1)
        for agent_i in range(self.num_agents):
            q_values[agent_i] = getattr(self, 'agent_{}'.format(agent_i))(x)

        return torch.cat(q_values, dim=1)


def soft_update(net, net_target, tau):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer, gamma, batch_size):
    state, action, reward, next_state, done_mask = memory.sample(batch_size)

    next_state_action_logits = mu_target(next_state)
    _, n_agents, action_size = next_state_action_logits.shape
    next_state_action_logits = next_state_action_logits.view(batch_size * n_agents, action_size)
    next_state_action = F.gumbel_softmax(logits=next_state_action_logits, tau=0.1, hard=True)
    next_state_action = next_state_action.view(batch_size, n_agents, action_size)

    target = reward + gamma * q_target(next_state, next_state_action) * done_mask
    q_loss = F.smooth_l1_loss(q(state, action), target.detach())
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()

    state_action_logits = mu(state)
    state_action_logits = state_action_logits.view(batch_size * n_agents, action_size)
    state_action = F.gumbel_softmax(logits=state_action_logits, tau=0.1, hard=True)
    state_action = state_action.view(batch_size, n_agents, action_size)

    mu_loss = -q(state, state_action).mean()  # That's all for the policy loss.
    q_optimizer.zero_grad()
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()


def test(env, num_episodes, mu):
    score = np.zeros(env.n_agents)
    for episode_i in range(num_episodes):
        state = env.reset()
        done = [False for _ in range(env.n_agents)]

        while not all(done):
            # env.render()
            action_logits = mu(torch.Tensor(state).unsqueeze(0))
            action = action_logits.argmax(dim=2).squeeze(0).data.cpu().numpy().tolist()
            next_state, reward, done, info = env.step(action)
            score += np.array(reward)
            state = next_state

    return sum(score / num_episodes)


def main(env_name, lr_mu, lr_q, tau, gamma, batch_size, buffer_limit, max_episodes, log_interval, test_episodes,
         warm_up_steps, update_iter, gumbel_max_temp, gumbel_min_temp):
    env = gym.make(env_name)
    test_env = gym.make(env_name)
    # test_env = Monitor(test_env, directory='recordings',
    #                   video_callable=lambda episode_id: episode_id % test_episodes == 0)
    memory = ReplayBuffer(buffer_limit)

    q, q_target = QNet(env.observation_space, env.action_space), QNet(env.observation_space, env.action_space)
    q_target.load_state_dict(q.state_dict())
    mu, mu_target = MuNet(env.observation_space, env.action_space), MuNet(env.observation_space, env.action_space)
    mu_target.load_state_dict(mu.state_dict())

    score = np.zeros(env.n_agents)

    mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
    q_optimizer = optim.Adam(q.parameters(), lr=lr_q)

    for episode_i in range(max_episodes):
        temperature = max(gumbel_min_temp,
                          gumbel_max_temp - (gumbel_max_temp - gumbel_min_temp) * (episode_i / (0.6 * max_episodes)))
        state = env.reset()
        done = [False for _ in range(env.n_agents)]
        step_i = 0
        while not all(done):
            action_logits = mu(torch.Tensor(state).unsqueeze(0))
            action_one_hot = F.gumbel_softmax(logits=action_logits.squeeze(0), tau=temperature, hard=True)
            action = torch.argmax(action_one_hot, dim=1).data.numpy()
            next_state, reward, done, info = env.step(action)
            step_i += 1
            if step_i >= env._max_steps or (step_i < env._max_steps and not all(done)):
                _done = [False for _ in done]
            else:
                _done = done
            memory.put((state, action_one_hot.data.numpy(), (np.array(reward)).tolist(), next_state,
                        np.array(_done, dtype=int).tolist()))
            score += np.array(reward)
            state = next_state

        if memory.size() > warm_up_steps:
            for i in range(update_iter):
                train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer, gamma, batch_size)
                soft_update(mu, mu_target, tau)
                soft_update(q, q_target, tau)

        if episode_i % log_interval == 0 and episode_i != 0:
            test_score = test(test_env, test_episodes, mu)
            print("#{:<10}/{} episodes , avg train score : {:.1f}, test score: {:.1f} n_buffer : {}"
                  .format(episode_i, max_episodes, sum(score / log_interval), test_score, memory.size()))
            if USE_WANDB:
                wandb.log({'episode': episode_i, 'test-score': test_score, 'gumbel_temperature': temperature,
                           'buffer-size': memory.size(), 'train-score': sum(score / log_interval)})
            score = np.zeros(env.n_agents)

    env.close()
    test_env.close()

    # save model to file
    if not os.path.exists("../models"): os.mkdir("../models")
    model_dir = '../models/'+env_name+'_maddpg_'+str(max_episodes)+'.pth'
    torch.save(mu.state_dict(), model_dir)
    print('Saved model q.state_dict() to ' + model_dir)

if __name__ == '__main__':
    # Lets gather arguments
    import argparse

    parser = argparse.ArgumentParser(description='Qmix')
    parser.add_argument('--env-name', required=False, default='ma_gym:Checkers-v0')
    parser.add_argument('--seed', type=int, default=1, required=False)
    parser.add_argument('--no-recurrent', action='store_true')
    parser.add_argument('--max-episodes', type=int, default=10000, required=False)

    # Process arguments
    args = parser.parse_args()

    kwargs = {'env_name': args.env_name,
              'lr_mu': 0.0005,
              'lr_q': 0.001,
              'batch_size': 32,
              'tau': 0.005,
              'gamma': 0.99,
              'buffer_limit': 50000,
              'log_interval': 20,
              'max_episodes': args.max_episodes,
              'test_episodes': 5,
              'warm_up_steps': 2000,
              'update_iter': 10,
              'gumbel_max_temp': 10,
              'gumbel_min_temp': 0.1}
    if USE_WANDB:
        import wandb

        wandb.init(project='minimal-marl', name=args.env_name+'_maddpg_'+str(args.max_episodes), config={'algo': 'maddpg', **kwargs, }, monitor_gym=True)

    main(**kwargs)
