from causal_rl.environments import SemEnv
import torch

env = SemEnv()

obs = env.reset()
done = False
t = 0
actions = [0,1,2,None]
for action in actions:
    obs, reward, done, _ = env.step(action)
    print(action, reward)
