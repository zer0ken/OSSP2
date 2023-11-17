
import numpy as np


class MultiAgentReplayBuffer:
    def __init__(self, max_size, critic_dims, actor_dims,
                n_actions, n_agents, batch_size):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.actor_dims = actor_dims
        
        self.state_memory = np.zeros((self.mem_size, critic_dims))
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)
        
        self.init_actor_memory()
        
    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []
        
        for i in range(self.n_agents):
            self.actor_state_memory.append(
                np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_new_state_memory.append(
                np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_action_memory.append(
                np.zeros((self.mem_size, self.n_actions[i])))
    
    def store_transition(self, raw_obs, state, action, reward, 
                        raw_new_obs, new_state, done):
        index = self.mem_cntr % self.mem_size
        
        for agent in range(self.n_agents):
            self.actor_state_memory[agent][index] = raw_obs[agent]
            self.actor_new_state_memory[agent][index] = raw_new_obs[agent]
            self.actor_action_memory[agent][index] = action[agent]
        
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1
    
    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        
        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        terminals = self.terminal_memory[batch]
        
        actor_states = []
        actor_new_states = []
        actions = []
        
        for agent in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent][batch])
            actor_new_states.append(self.actor_new_state_memory[agent][batch])
            actions.append(self.actor_action_memory[agent][batch])
        
        return actor_states, states, actions, rewards,\
            actor_new_states, new_states, terminals
    
    def ready(self):
        return self.mem_cntr >= self.batch_size