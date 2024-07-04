import gym
import numpy as np

def patched_env_step_passive_checker(env, action):
    observation, reward, terminated, truncated, info = env.step(action)

    # Check the return of terminated and truncated
    if not isinstance(terminated, (bool,)):
        raise TypeError(f"Expected terminated to be a boolean, got {type(terminated)}")
    if not isinstance(truncated, (bool,)):
        raise TypeError(f"Expected truncated to be a boolean, got {type(truncated)}")

    return observation, reward, terminated, truncated, info

class CustomGymWrapper(gym.Wrapper):
    def step(self, action):
        return patched_env_step_passive_checker(self.env, action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

# Create the custom environment
env = CustomGymWrapper(gym.make('CartPole-v1'))
