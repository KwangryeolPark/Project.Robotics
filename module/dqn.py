import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

class DQN(object):
    def __init__(self, state_size, action_size, cfg):
        self.dqn = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, action_size)
        )
        self.optimizer = optim.Adam(self.dqn.parameters())
        self.steps_done = 0
        self.memory = deque(maxlen=10_000)
        self.cfg = cfg
        self.dqn = self.dqn.to(self.cfg['device'])
        
        assert 'batch_size' in list(self.cfg.keys())
        assert 'gamma' in list(self.cfg.keys())
    
    def memorize(self, state, action, reward, next_state):
        self.memory.append((state, action, torch.FloatTensor([reward]), torch.FloatTensor([next_state])))
    
    def act(self, state):
        eps_threshold = 0.05 + (0.9 - 0.05) * math.exp(-1 * self.steps_done / 200)
        self.steps_done += 1
        
        state = state.to(self.cfg['device'])
        if random.random() > eps_threshold:
            return self.dqn(state).data.max(1)[1].view(1, 1)
        else:
            return torch.LongTensor([[random.randrange(2)]]).to(self.cfg['device'])
    
    def learn(self):
        if len(self.memory) < self.cfg['batch_size']:
            return
        
        batch = random.sample(self.memory, self.cfg['batch_size'])
        states, actions, rewards, next_states = zip(*batch)
        
        states = torch.cat(states).to(self.cfg['device'])
        actions = torch.cat(actions).to(self.cfg['device'])
        rewards = torch.cat(rewards).to(self.cfg['device'])
        next_states = torch.cat(next_states).to(self.cfg['device'])
        
        current_q = self.dqn(states).gather(1, actions)
        
        max_next_q = self.dqn(next_states).detach().max(1)[0]
        expected_q = rewards + (self.cfg['gamma'] * max_next_q)
        
        loss = F.mse_loss(current_q.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()