from causal_rl.environments import SemEnv
import torch

env = SemEnv()

obs = env.reset()
done = False
t = 0
actions = [0,1,2,None]
while not done:
    obs, reward, done, _ = env.step(actions[t % len(actions)])
    t += 1
    print(actions[t % len(actions)], reward)
