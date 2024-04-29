import gym
import slimevolleygym
import numpy as np
from time import sleep
import imageio
import neat 


class Simulate():
    def __init__(self, env_name='SlimeVolley-v0', model=None, num_timesteps=50, naive: bool = False, verbose: bool = False):
        """
        Model should have a predict function which takes in 12-vector of observatiosn and returns a 3-vector action. 

        """
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.num_timesteps = num_timesteps
        self.done = False
        self.step = 0
        self.model = model
        self.naive = False
        self.naive = naive
        self.verbose = verbose
        self.frames = []

        if not self.naive and self.model is None:
            raise ValueError("A model must be provided unless 'naive' is set to True.")

    def run_simulation(self):
        self.frames = []
        self.done = False
        self.step = 0
        obs = self.env.reset() 

        while not self.done and self.step < self.num_timesteps:

            if self.naive:
                action = self.env.action_space.sample()
            else:
                action = self.model.predict(obs, self.step)

            obs, reward, done, info = self.env.step(action)

            if self.verbose:
                print(f"Step {self.step}: Observation = {obs}, Reward = {reward}, Done = {done}")

            frame = self.env.render(mode='rgb_array')
            self.frames.append(frame)

            if done:
                obs = self.env.reset()
                self.done = True

            self.step += 1

        self.env.close()

    def fitness_function(self):
        pass

    def visualise_genome(self):
        pass

    def save_simulation(self, filename='sim.gif'):
        imageio.mimsave(filename, self.frames, fps=30)
        print(f"Simulation saved to {filename}")

def simulate_environment_2(env, model, num_steps=1000):
    observation = env.reset()
    done = False
    step = 0

    while not done and step < num_steps:
        # Generate a random action: Assuming each action dimension is continuous
        # You may need to adjust the range depending on the environment's specifications
        action = model.predict(observation, step) 
        
        # Take the action in the environment
        observation, reward, done, info = env.step(action)
        
        # Output the results to observe what's happening
        print(f"Step {step}: Observation = {observation}, Reward = {reward}, Done = {done}")
        
        step += 1

    # Close the environment when done
    env.close()

if __name__ == "__main__":
    naive_sim = Simulate(naive=True, num_timesteps=1000)
    naive_sim.run_simulation()
    naive_sim.save_simulation()
