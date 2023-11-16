import torch as T
import torch.nn.functional as F

from Agent import Agent


class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions,
                scenario='simple', 
                alpha=0.01, beta=0.01, fc1=64, fc2=64,
                gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/'):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario
        
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims, critic_dims, n_agents, n_actions,
                                    agent_idx, chkpt_dir, 
                                    alpha, beta, gamma, tau, fc1, fc2))
        
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()
    
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()
    
    def choose_action(self, raw_obs):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx])
            actions.append(action)
        
        return actions

    def learn(self, memory):
        if not memory.ready():
            return
        
        actor_states, states, actions, rewards, \
            new_actor_states, new_states, dones = memory.sample_buffer()
        
        device = self.agents[0].actor.device
        
        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards).to(device)
        new_states = T.tensor(new_states, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)
        
        agent_actions = []
        new_agent_target_actions = []
        reconsidered_agent_actions = []
        
        for agent_idx, agent in enumerate(self.agents):
            agent_actions.append(actions[agent_idx])
            
            new_local_states = T.Tensor(new_actor_states[agent_idx], dtype=T.float).to(device)
            new_target_action = agent.target_actor.forward(new_local_states)
            new_agent_target_actions.append(new_target_action)
            
            actor_state = T.Tensor(actor_states[agent_idx], dtype=T.float).to(device)
            new_action = agent.actor.forward(actor_state)
            reconsidered_agent_actions.append(new_action)
        
        # actions that target actor would take on next_state
        new_target_actions = T.cat([ta for ta in new_agent_target_actions], dim=1)
        
        # actions that current actor would take on sampled state
        reconsidered_actions = T.cat([ra for ra in reconsidered_agent_actions], dim=1)
        
        # substantially the same as `actions` but in different shape? IDK :P
        # action that actually/already taken by actor on each sampled state
        old_actions = T.cat([a for a in agent_actions], dim=1)
        
        for agent_idx, agent in enumerate(self.agents):
            new_target_critic = agent.target_critic.forward(new_states, new_target_actions)
            new_target_critic[dones[:, 0]] = 0.0
            critic = agent.critic.forward(states, old_actions.flatten())
            
            # `target` denotes `y^`; expected value of correct critic `y`
            # refer to eq (8)
            target = rewards[:, agent_idx] + agent.gamma*new_target_critic
            critic_loss = F.mse_loss(target, critic)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()
            
            actor_loss = agent.critic.forward(states, reconsidered_actions).flatten()
            actor_loss = -T.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()
            
            agent.update_network_parameters()