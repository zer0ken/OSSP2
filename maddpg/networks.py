import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from SavableModule import SavableModule

class CriticNetwork(SavableModule):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims,
                n_agents, n_actions, name, chkpt_dir,
                *args, **kwargs):
        super().__init__(name, chkpt_dir, *args, **kwargs)
        
        print(name, input_dims, fc1_dims, fc2_dims)
        
        self.fc1 = nn.Linear(input_dims + n_actions*n_agents, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state, action):
        x = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)
        
        return q
    
class ActorNetwork(SavableModule):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                n_actions, name, chkpt_dir, 
                *args, **kwargs):
        super().__init__(name, chkpt_dir, *args, **kwargs)

        print(name, input_dims, fc1_dims, fc2_dims)

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = T.softmax(self.pi(x), dim=1)

        return pi