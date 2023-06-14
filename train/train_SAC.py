#https://github.com/WSU-Data-Science/Robot-Arm-RL/tree/a93b4aae06e280c0971b25dfcfec0f1b294092b6/articulations-robot-demo-master
#https://github.com/tarod13/SAC
#https://docs.cleanrl.dev/rl-algorithms/sac/
#https://spinningup.openai.com/en/latest/algorithms/sac.html
#https://www.youtube.com/watch?v=ioidsRlf79o
#https://medium.com/intro-to-artificial-intelligence/soft-actor-critic-reinforcement-learning-algorithm-1934a2c3087f
#https://towardsdatascience.com/double-deep-q-networks-905dd8325412

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import torch
import gymnasium as gym
from utility.utility import Network,Memory
from torch import nn,optim
import argparse
#Information unit
Rollout = namedtuple('Rollout', ['state', 'action', 'reward', 'next', 'done'])

class SACAgent:
    def __init__(self, env, lr=1e-3, gamma=0.99, soft_update_tau=0.01,
                 memory_size=2000, hidden_size=64, log_std_range=[-20,2]):
        # Initialise dimensions and learning parameters
        self.env = env
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]
        self.lr = lr
        self.gamma = gamma
        # tau -> (0,1) controls 'how much the target models' weights will be updated toward the base models' weights
        self.tau = soft_update_tau

        self.hidden_size = hidden_size
        # range of log_std value
        self.min_clamp = log_std_range[0]
        self.max_clamp = log_std_range[-1]


        # Initialise networks/criterion
        self._initialise_model()
        self.update_target_networks(tau=1)

        self.criterion = nn.MSELoss()
        
        # Initialise Replay Memory
        self.memory = Memory(capacity=memory_size)
        
    def _initialise_model(self):
        # Critic networks and their copies

        # n_states+n_actions -> we want to have information about the state and action
        self.critic1 = Network(self.n_states + self.n_actions, 1, self.hidden_size)
        self.critic2 = Network(self.n_states + self.n_actions, 1, self.hidden_size)
        self.target_critic1 = Network(self.n_states + self.n_actions, 1, self.hidden_size)
        self.target_critic2 = Network(self.n_states + self.n_actions, 1, self.hidden_size)
        
        # Actor network
        # n_actions *2 -> we want to have mean and std (for Normal distribution)
        self.actor = Network(self.n_states, self.n_actions*2, self.hidden_size)
        
        # Temperature (alpha) -> -log(P(x))
        self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape)).item()
        # Alpha ->  Entropy regularization coefficient
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp()
        
        # Optimizer
        self.critic_optim1 = optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.critic_optim2 = optim.Adam(self.critic2.parameters(), lr=self.lr)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.lr, eps=1e-4)
        
    def update_target_networks(self, tau=None):
        # Initialize tau if None
        if tau is None:
            tau = self.tau
        # Update netrowks.parameters based on 'soft target update'
        for local_param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(tau * local_param.data + (1-tau) * target_param.data)
            
        for local_param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(tau * local_param.data + (1-tau) * target_param.data)
        
    def get_action_prob(self, state, epsilon=1e-6):
        # Reparametrization trick -> https://deepganteam.medium.com/basic-policy-gradients-with-the-reparameterization-trick-24312c7dbcd

        state = state.float()
        output = self.actor(state)
        # Get mean and standard deviation for normal distribution 
        mean, log_std = output[..., :self.n_actions], output[..., self.n_actions:]
        log_std = torch.clamp(log_std, self.min_clamp, self.max_clamp)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z)
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob
    
    def critic_loss(self, states, actions, rewards, nextstates, done):
        
        with torch.no_grad():
            next_actions, next_log_probs = self.get_action_prob(nextstates)
            next_q1 = self.target_critic1(torch.cat((nextstates, next_actions), dim=1))
            next_q2 = self.target_critic2(torch.cat((nextstates, next_actions), dim=1))
            min_next_q = torch.min(next_q1, next_q2)
            # https://spinningup.openai.com/en/latest/algorithms/sac.html#key-equations -> minQ -alpha*log_std
            soft_state = min_next_q - self.alpha * next_log_probs
            # if failed add extra value 
            target_q = rewards + (1 - done) * self.gamma * soft_state
            
        pred_q1 = self.critic1(torch.cat((states, actions), dim=1))
        pred_q2 = self.critic2(torch.cat((states, actions), dim=1))

        loss1 = self.criterion(pred_q1, target_q)
        loss2 = self.criterion(pred_q2, target_q)

        return loss1, loss2
        
        
    def actor_loss(self, states):
        actions, log_prob = self.get_action_prob(states)
        q_values1 = self.critic1(torch.cat((states, actions), dim=1))
        q_values2 = self.critic2(torch.cat((states, actions), dim=1))
        
        min_q_values = torch.min(q_values1, q_values2)
        # minq-alpha
        policy_loss = -(min_q_values-self.alpha * log_prob).mean()

        
        return policy_loss, log_prob
    
    # https://towardsdatascience.com/entropy-regularization-in-reinforcement-learning-a6fa6d7598df
    def temperature_loss(self, log_prob):
        loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        return loss
        
    def _choose_action(self, state, random=False):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state)
            
        if random:
            actions = self.env.action_space.sample()            
       
        else:
            with torch.no_grad():
                actions, _ = self.get_action_prob(state)
                
        actions = np.array(actions).reshape(-1) 
        return actions
        

    # get information from batch
    def unpack_batch(self, samples):
        batch_data = list(map(list, zip(*samples)))

        batch_states = torch.tensor(np.array(batch_data[0])).float()
        batch_actions = torch.tensor(np.array(batch_data[1])).float()
        batch_rewards = torch.tensor(np.array(batch_data[2])).float().unsqueeze(-1)
        batch_nextstates = torch.tensor(np.array(batch_data[3])).float()
        batch_done = torch.tensor(np.array(batch_data[4])).float().unsqueeze(-1)
        
        return batch_states, batch_actions, batch_rewards, batch_nextstates, batch_done

    def learn(self, samples):
        batch_states, batch_actions, batch_rewards, batch_nextstates, batch_done = self.unpack_batch(samples)
        
        self.critic_optim1.zero_grad()
        self.critic_optim2.zero_grad()
        critic_loss1, critic_loss2 = self.critic_loss(
            batch_states, batch_actions, batch_rewards, batch_nextstates, batch_done)
        critic_loss1.backward()
        critic_loss2.backward()
        self.critic_optim1.step()
        self.critic_optim2.step()

        self.actor_optim.zero_grad()         
        actor_loss, log_probs = self.actor_loss(batch_states)
        actor_loss.backward()
        self.actor_optim.step()

        self.alpha_optim.zero_grad()               
        alpha_loss = self.temperature_loss(log_probs)
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()
        
        self.update_target_networks(tau=self.tau)
        return torch.min(critic_loss1, critic_loss2).item(), actor_loss.item(), alpha_loss.item()
    

    
    def train(self, n_episode=250, initial_memory=None,
              report_freq=10, batch_size=32):
        # Initialize memory with random samples
        if initial_memory is None:
            initial_memory = batch_size*4
        
        self.memory.initialise_memory(self.env,size=initial_memory)
        results = []
        for i in range(n_episode):
            state, _ = self.env.reset()
            done, truncated = False, False
            eps_reward = 0

            while not (done or truncated):
                
                action = self._choose_action(state)
                nextstate, reward, done, truncated, _ = self.env.step(action)
                roll = Rollout(state, action, reward, nextstate, done)
                self.memory.store_rollout(roll)
                state = nextstate
                # get random sample 
                samples = self.memory.sample_batch(batch_size)
                
                critic_loss, actor_loss, alpha_loss = self.learn(samples)

                
                eps_reward += reward
            results.append(eps_reward)
                
            # Display progress
            if i % report_freq == 0:
                print(f'Episode {i}/{n_episode} \t Reward: {eps_reward:.4f} \t Critic Loss: {critic_loss:.3f}\t '+
                      f'Actor Loss: {actor_loss:.3f}\t Alpha Loss: {alpha_loss:.3f}\t Alpha: {self.alpha.item():.4f}')
        return results
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str,help="name of the environment e.g 'Reacher-v4'",default='Reacher-v4',required=False)
    parser.add_argument('--epos', type=int,help="number of epochs",default=1000,required=False)
    

    args= parser.parse_args()
    env_name = args.env
    epos = args.epos

    env = gym.make(env_name)
    agent = SACAgent(env, lr=3e-4, gamma=0.99, memory_size=5000, hidden_size=256)
    learning_data = agent.train(n_episode=epos, batch_size=200, report_freq=10)
    algo = 'gym_'+env_name.replace('-','_')+'_'
    actor = os.path.join('models/actor_'+algo+'_'+str(epos)+
                         '.pt')
    torch.save(agent.actor.state_dict(),actor)

    plt.plot(learning_data,label='Reward over episodes')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()