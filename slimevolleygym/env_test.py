import gym
import slimevolleygym
import numpy as np
from time import sleep
import imageio
import neat 



class Simulate():
    def __init__(self, env_name='SlimeVolley-v0', model=None, num_timesteps=50, naive: bool = False, verbose: bool = False):
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

    def save_simulation(self, filename='sim.gif'):
        imageio.mimsave(filename, self.frames, fps=30)
        print(f"Simulation saved to {filename}")



def simulate_environment(env_name, num_timesteps):
    # Initialize the environment
    env = gym.make(env_name)
    observation = env.reset()
    
    # To store results
    results = []

    # Evolve the environment over the given number of timesteps
    for _ in range(num_timesteps):
        # Take a random action
        action = env.action_space.sample()
        # Step the environment
        observation, reward, done, info = env.step(action)
        
        # Store the output
        results.append({
            'observation': observation,
            'reward': reward,
            'done': done,
            'info': info
        })

        # Check if the episode is done and reset if true
        if done:
            observation = env.reset()

    env.close()
    return results

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

import numpy as np

class SimpleModel:
    def __init__(self, action_size, fixed_action=None):
        self.action_size = action_size
        # Set a default action if none provided
#        if fixed_action is None:
            # This can be adjusted depending on the expected action range and size
            # [1, 0, 0] -> Move left
            # [0, 1, 0] -> Move right
            # [0, 0, 1] -> Jump
            # [1, 1, 0] -> Nothing? -> Move left + move right = no movement
            # [0, 1, 1] -> Jump & move right
            # [1, 0, 1] -> Jump & move left 
            # [1, 1, 1] -> Jump 
            # [-1, 0, 0] -> Nothing? Probably the same for all out of range [0, 1]...
            # [1/2, 0, 0] -> Move left (possible with leff velocity? Probably...)

            # Random action
#            self.fixed_action = np.random.randint(0, 2, 3) #-> Leads to single direction movement
#            self.fixed_action = np.random.uniform(-1, 1, 3)
           

#        else:
#            self.fixed_action = fixed_action

    def predict(self, observation, step):
        # Ignores the observation and returns the fixed action
        if step<=30:
            self.fixed_action = [1, 0, 0]
        elif step>=31 and step<=70:
            self.fixed_action = [0, 1, 0]
        else:
            self.fixed_action = [0, 0, 1]
 
        return self.fixed_action




# Use the function
env_name = "SlimeVolley-v0"
model = SimpleModel(3)
num_timesteps = 100      # Number of timesteps to simulate
outputs = simulate_environment_2(gym.make(env_name), model, num_timesteps)



# Printing the outputs
#for i, output in enumerate(outputs):
#    print(f"Step {i+1}: Observation = {output['observation']}, Reward = {output['reward']}, Done = {output['done']}")

def main():
        # Initialize Gym environment
    env = gym.make("SlimeVolley-v0")
    env.seed(np.random.randint(0, 10000))  # For reproducibility

    frames = []

    step = 0

    # Initialize the simple model
    policy = SimpleModel(action_size=3, step=step)  # Adjust based on actual environment

#    # Rendering mode
#    RENDER_MODE = True
#    if RENDER_MODE:
#        env.render()

    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        step += 1
        # Get action from the simple model
        action = policy.predict(obs, step=step)

        # Step the environment
        obs, reward, done, info = env.step(action)
        total_reward += reward

#        # Render the environment if rendering is enabled
#        if RENDER_MODE:
#            env.render()
#            sleep(0.02)  # Control the speed of rendering
#       Run with xvfb-run -s "-screen 0 1400x900x24" over ssh to get around no graphical display.
#       Will save as gif
        frame = env.render(mode='rgb_array')
        frames.append(frame)
    print(f"Total Reward: {total_reward}")
    env.close()
    imageio.mimsave('simulation.gif', frames, fps=30)


if __name__ == "__main__":
    naive_sim = Simulate(naive=True, num_timesteps=1000)
    naive_sim.run_simulation()
    naive_sim.save_simulation()
