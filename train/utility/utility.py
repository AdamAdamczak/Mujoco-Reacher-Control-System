import random
from collections import deque
from torch import nn


class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def store_rollout(self, rollout):
        self.memory.append(rollout)

    def sample_batch(self, batch_size):
        return random.sample(self.memory, batch_size)

    def initialise_memory(self, env, size):
        state,_ = env.reset()
        for _ in range(size):
            action = env.action_space.sample()
            next_state, reward, done,truncated, _ = env.step(action)
            rollout = (state, action, reward, next_state, done)
            self.store_rollout(rollout)
            if done:
                state,_ = env.reset()
            else:
                state = next_state




class Network(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(Network, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )
        
    def forward(self, x):
        x = self.network(x)
        return x
    