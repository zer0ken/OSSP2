import numpy as np

from pettingzoo.mpe import simple_adversary_v3

from MADDPG import MADDPG
from MultiAgentReplayBuffer import MultiAgentReplayBuffer

def observations_to_state_vector(observations):
    state = np.array([])
    for obs in observations:
        state = np.concatenate((state, obs))
    return state

def dict_to_list(dict_, agent_names):
    list_ = []
    for agent_name in agent_names:
        list_.append(dict_[agent_name])
    return list_

if __name__ == '__main__':
    PRINT_INTERVAL = 500
    N_GAMES = 30000
    N_AGENTS = 3
    MAX_STEPS = 25
    
    evaluate = False
    
    scenario = 'simple'
    env = simple_adversary_v3.parallel_env(render_mode='human', max_cycles=MAX_STEPS, N=N_AGENTS-1, continuous_actions=True)
    observation, _ = env.reset()
    
    env.render()
    
    n_agents = env.num_agents
    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(env.observation_space(env.agents[i]).shape[0])
    critic_dims = sum(actor_dims)
    
    # todo: check if the env has attribute 'n'
    # n_actions = env.action_spaces[0].shape[0]
    n_actions = []
    for i in range(n_agents):
        n_actions.append(env.action_space(env.agents[i]).shape[0])
    
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, scenario,
                            alpha=0.01, beta=0.01, fc1=64, fc2=64)
    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, n_actions, n_agents, 128)
    
    total_steps = 0
    score_history = []
    best_score = -np.inf
    
    if evaluate:
        maddpg_agents.load_checkpoint()
    
    for i in range(N_GAMES):
        score = 0
        done = [False] * n_agents
        episode_step = 0
        observation, _ = env.reset()
        observation = dict_to_list(observation, env.agents)
                
        while not any(done):
            actions = maddpg_agents.choose_action(observation)
            actions_dict = {agent: action for agent, action in zip(env.agents, actions)}
            
            new_observation, reward, done, truncated, info = env.step(actions_dict)
            new_observation, reward, done = map(lambda x: dict_to_list(x, env.agents), [new_observation, reward, done])
            
            state = observations_to_state_vector(observation)
            new_state = observations_to_state_vector(new_observation)
            
            memory.store_transition(observation, state, actions, reward,
                                    new_observation, new_state, done)
            
            if total_steps % 100 == 0 and not evaluate:
                maddpg_agents.learn(memory)
            
            observation = new_observation
            
            score += sum(reward)
            total_steps += 1
            episode_step += 1
        
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        
        if not evaluate:
            if avg_score > best_score:
                best_score = avg_score
                maddpg_agents.save_checkpoint()
        if i % PRINT_INTERVAL == 0 and i > 0:
            print(f'episode {i} | score: {score} | avg score: {avg_score:.2f}')