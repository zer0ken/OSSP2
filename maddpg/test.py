from time import sleep
from pettingzoo.mpe import simple_adversary_v3
env = simple_adversary_v3.env(render_mode='human')

for i in range(100):
    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            action = env.action_space(agent).sample() # this is where you would insert your policy

        env.step(action)
        sleep(0.1)
env.close()