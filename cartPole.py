# -*- coding: utf-8 -*-

import gym
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque


env = gym.make('CartPole-v1')


device = torch.device('cuda')
dtype = torch.float

class Q_net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Q_net, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, H)
        self.linear3 = nn.Linear(H, D_out)

    def forward(self, x):
        h1 = nn.functional.relu(self.linear1(x))
        h2 = nn.functional.relu(self.linear2(h1))
        y = self.linear3(h2)
        return y


class Memory():
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experence):
        self.buffer.append(experence)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[i] for i in idx]


def train_dqn_model(episodes, max_steps, gamma, hidden_size, memory_size, batch_size, learning_rate):
    model = Q_net(4, hidden_size, 2)

    loss_fn = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    memory = Memory(max_size=memory_size)

    env.reset()
    state, reward, done, _ = env.step(env.action_space.sample())

    memory = Memory(max_size=memory_size)

    for i in range(batch_size):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        if done:
            next_state = np.zeros(state.shape)
            memory.add((state, action, reward, next_state))
            env.reset()
            state, reward, done, _ = env.step(env.action_space.sample())
        else:
            memory.add((state, action, reward, next_state))
            state = next_state

    rewards_list = []

    for ep in tqdm(range(episodes)):
        state = env.reset()
        epsilon = 1 - 1.25 * (ep / episodes)
        epsilon = np.clip(epsilon, 0.01, 0.9)
        reward_all = 0
        for step in range(max_steps):
            ep_done = False
            #env.render()
            if np.random.rand(1) < epsilon:
                action = env.action_space.sample()
            else:
                input_s = torch.FloatTensor(state.reshape((1, *state.shape)))
                Q_s = model(input_s)
                action = torch.argmax(Q_s).detach().numpy()
            next_state, reward, done, _ = env.step(action)
            reward_all += reward

            if done:
                next_state = np.zeros(state.shape)
                ep_done = True
                memory.add((state, action, reward, next_state))

                env.reset()
                state, reward, done, _ = env.step(env.action_space.sample())
            else:
                memory.add((state, action, reward, next_state))
                state = next_state

            batch = memory.sample(batch_size=batch_size)
            states = np.array([each[0] for each in batch])
            actions = np.array([each[1] for each in batch])
            rewards = np.array([each[2] for each in batch])
            next_states = np.array([each[3] for each in batch])

            predict_Q = model(torch.FloatTensor(states))
            next_Q = model(torch.FloatTensor(next_states)).detach().numpy()
            ep_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
            next_Q[ep_ends] = (0, 0)

            target_Q = predict_Q.clone().detach().numpy()
            targets = rewards + gamma * np.max(next_Q, axis=1)

            actions = torch.LongTensor(actions.reshape(actions.size, 1))
            one_hot = torch.zeros(batch_size, 2).scatter_(1, actions, 1)
            predict_Q = torch.mul(predict_Q, one_hot).sum(dim=1)

            optimizer.zero_grad()
            loss = loss_fn(torch.FloatTensor(targets), predict_Q)
            loss.backward()
            optimizer.step()

            if ep_done or step == max_steps-1:
                rewards_list.append((ep, reward_all))
                if ep % 10 == 0:
                    pass
                    #print('Episode: {}'.format(ep),
                    #      'Total reward: {}'.format(reward_all),
                    #      'Training loss: {:.4f}'.format(loss/batch_size),
                    #      'Explore P: {:.4f}'.format(epsilon))
                break
    return model, rewards_list


def test_dqn_model(model, episodes, max_steps):
    env.reset()
    rewards_list = []
    state, reward, done, _ = env.step(env.action_space.sample())
    for ep in range(episodes):
        rewards = 0
        for step in range(max_steps):
            env.render()
            Qs = model(torch.FloatTensor(state.reshape((1, *state.shape))))
            action = torch.argmax(Qs).detach().cpu().numpy()

            next_state, reward, done, _ = env.step(action)
            rewards += reward
            if done or step == max_steps-1:
                env.reset()
                state, reward, done, _ = env.step(env.action_space.sample())
                rewards_list.append(rewards)
                break
            else:
                state = next_state
    return rewards_list


if __name__ == '__main__':
    train_episodes = 1000
    max_steps = 200
    gamma = 0.99

    hidden_size = 64
    learning_rate = 0.0001
    memory_size = 10000
    batch_size = 20

    test_episodes = 10
    test_max_steps = 400

    dqn_model, rewards_list = train_dqn_model(train_episodes, max_steps, gamma, hidden_size, memory_size, batch_size, learning_rate)

    test_rewards_list = test_dqn_model(dqn_model, test_episodes, test_max_steps)

    eps, rewards = np.array(rewards_list).T
    smoothed_rewards = np.cumsum(np.insert(rewards, 0, 0))
    smoothed_rewards = (smoothed_rewards[10:] -smoothed_rewards[:-10]) / 10
    plt.plot(eps[-len(smoothed_rewards):], smoothed_rewards)
    plt.plot(eps, rewards, color='grey', alpha=0.3)
    plt.show()

    print(test_rewards_list)
    env.close()
