import gym
import numpy as np
from Agent import Agent


if __name__ == '__main__':
    env = gym.make('Pendulum-v1', g=9.81, render_mode='human')
    
    agent = Agent(input_dims=env.observation_space.shape, env=env,
                    n_actions=env.action_space.shape[0], noise=0.1, gamma=0.99)
    
    n_games = 250

    figure_file = 'plots/pendulum.png'

    best_score = env.reward_range[0]
    score_history = []

    """
    config
    """
    load_checkpoint = True
    evaluate = False
    do_punch = True

    if load_checkpoint:
        n_steps = 0
        while n_steps <= agent.batch_size:
            observation, _ = env.reset()
            action = env.action_space.sample()
            new_observation, reward, done, truncated, info = env.step(action)
            agent.remember(observation, action, reward, new_observation, done)
            n_steps += 1
        agent.learn()
        agent.load_models()
    
    env.reset()
    env.render()

    for i in range(n_games):
        observation, _ = env.reset()
        done = False
        score = 0.0
        frame = 0
        last_score = 0.0
        while not done:
            frame += 1
            action = agent.choose_action(observation, evaluate)
            new_observation, reward, done, truncated, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, new_observation, done)
            if not evaluate:
                agent.learn()
            observation = new_observation
            if frame % 100 == 0:
                print(f'frame {frame} score_diff {score - last_score:.6f}')
                if do_punch and round(score - last_score) == 0:
                    punch = np.array([np.random.rand() * 4.0 - 2.0], dtype=np.float32)
                    for _ in range(np.random.randint(1, 5)):
                        env.step(punch)
                        print(f'\tpunch {punch[0]}')
                last_score = score
            if frame >= 5000:
                done = True
        score_history.append(score)
        avg_score = np.mean(score_history[-100:], dtype=np.float32)

        if avg_score > best_score:
            best_score = avg_score

        if not evaluate:
            agent.save_models()
        
        print(f'episode {i} score {score:.1f} avg score {avg_score:.1f}')