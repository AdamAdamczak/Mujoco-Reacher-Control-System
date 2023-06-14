import numpy as np
import torch
import gymnasium as gym
from train.utility.utility import Network
import argparse
class SACAgentEvaluator:
    def __init__(self, env, model_path,random,n_episode):
        self.env = env
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]
        self.n_episode=n_episode
        self.random = random
        self.model = Network(self.n_states, self.n_actions*2, hidden_dim=256)
        if not self.random:
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
    
    def choose_action(self, state):
        if self.random:
            action = self.env.action_space.sample()    
        else:
            state = torch.tensor(state).float()
            with torch.no_grad():
                output = self.model(state)
                mean, log_std = output[..., :self.n_actions], output[..., self.n_actions:]
                std = log_std.exp()
                normal = torch.distributions.Normal(mean, std)
                z = normal.sample()
                action = torch.tanh(z)
            action = action.detach().numpy()
        return action
    
    def evaluate(self):
        for i in range(self.n_episode):
            state, _ = self.env.reset()
            done, truncated = False, False
            eps_reward = 0
            n_steps = 0
            while not (done or truncated):
                action = self.choose_action(state)
                nextstate, reward, done, truncated, _ = self.env.step(action)
                state = nextstate
                eps_reward += reward
                n_steps += 1
                
        

        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,help="path to model (.pt)",default="models/model.pt",required=False)
    parser.add_argument('--env', type=str,help="name of the environment e.g 'Reacher-v4'",default='Reacher-v4',required=False)
    parser.add_argument('--rand',type=bool,help="random sample",default=False,required=False)
    parser.add_argument('--epos', type=int,help="number of epochs",default=50,required=False)

    
    args= parser.parse_args()
    model_path = args.model
    env_name = args.env
    random = args.rand
    epos = args.epos
    
    env = gym.make(env_name,render_mode ='human')
    agent_evaluator = SACAgentEvaluator(env, model_path,random,epos)
    agent_evaluator.evaluate()

if __name__ == '__main__':
    main()



