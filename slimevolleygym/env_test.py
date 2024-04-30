import gym
import slimevolleygym
import numpy as np
from time import sleep
import imageio
import neat 
import visualise
import datetime


def get_formatted_datetime():
    # Get the current datetime
    now = datetime.datetime.now()

    # Format the datetime as minutes_hours_day_month
    return now.strftime("%M_%H_%d_%m")


class Simulate:
    def __init__(self, config_path, env_name='SlimeVolley-v0', model=None, num_timesteps=50, naive: bool = False, verbose: bool = False):
        """
        Model should have a predict function which takes in 12-vector of observatiosn and returns a 3-vector action. 

        """
        self.timestamp = str(datetime.datetime.now().strftime("%M_%H_%d_%m"))

        self.step = 0
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

#        if not self.naive and self.model is None:
#            raise ValueError("A model must be provided unless 'naive' is set to True.")

        # NEAT Initialisation
        self.generations = 1000
        self.config_path = config_path
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             self.config_path
                             )
        self.pop = neat.Population(self.config)
        self.pop.add_reporter(neat.StdOutReporter(True))
        self.stats = neat.StatisticsReporter()
        self.pop.add_reporter(self.stats)
        self.node_names = {
                2: 'jump',
                1: 'right',
                0: 'left',
                -1: 'x_agenr',
                -2: 'y_agent',
                -3: 'xv_agent',
                -4: 'yv_agent',
                -5: 'x_ball',
                -6: 'y_ball',
                -7: 'xv_ball',
                -8: 'yv_ball',
                -9: 'x_opponent',
                -10: 'y_opponent',
                -11: 'xv_opponent',
                -12: 'yv_opponent',
        }
    def _fitness_function(self, observations, reward, info):
        """
        Fitness function to be maximised.

        """
        x, y, xv, yv, xb, yb, xvb, yvb, xo, yo, xov, yov = observations
        position = (x, y)
        velocity = (xv, yv)
        ball_position = (xb, yb)
        ball_velocity = (xvb, yvb)
        opp_position = (xo, yo)
        opp_velocity = (xov, yov)

        pass


    def _evaluate_geoneomes(self, genomes, config):
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)

            obs = self.env.reset()
            total_reward = 0

            visited_positions = set()
            prev_action = None


            for _ in range(self.num_timesteps):
                action = net.activate(obs)
                obs, reward, done, info = self.env.step(action)

                x, y, xv, yv, xb, yb, xvb, yvb, xo, yo, xov, yov = obs

                 # Fitness
                if reward == 1:
                     total_reward += 5
                elif reward == -1:
                    total_reward -= 5
                elif reward == 0:
                    total_reward -= 0.1

                total_reward -= 0.1 * abs(x - xb)
                total_reward -= 0.1 * abs(y - yb)

                if x == xb:
                    if y == yb:
                        total_reward += 3
                
                if x not in visited_positions:
                    total_reward += 1
                    visited_positions.add(x)

                if prev_action is not None:
                    action_change_penalty = sum(abs(action - prev_action))
                    total_reward -= action_change_penalty
                # End of Fitness
    
                if self.verbose:
                    print(f"Step {_}: Observation = {obs}, Reward = {reward}, Done = {done}")
                if done:
                    break

            genome.fitness = total_reward

    def generate_neat_network(self):
        """
        Use NEAT to produce a winning genome based on the SlimeBall fitness function. 

        """
        winner = self.pop.run(self._evaluate_geoneomes, self.generations)
        return winner

    def plot_genome(self, genome, filename, prune_unused=False):
        """
        Plots the given genome using the provided NEAT configuration.
    
        Args:
            config (neat.Config): The NEAT configuration object.
            genome (neat.DefaultGenome): The genome to be visualized.
            filename (str): Base filename to save the plots.
            node_names (dict): Dictionary mapping node identifiers to names.
            prune_unused (bool): Flag to determine whether to prune unused nodes.
        """
        filename = f"{self.timestamp}_{filename}" 
        # Draw and save the full network diagram
        visualise.draw_net(self.config, genome, False, node_names=self.node_names, filename=f'{filename}_network.svg')
        
        # Optionally draw and save the pruned network diagram
        if prune_unused:
            visualise.draw_net(self.config, genome, False, node_names=self.node_names, prune_unused=prune_unused, filename=f'{filename}_pruned_network.svg')

        # Save statistics plots (considering they are part of the genome's evaluation context)
        visualise.plot_stats(self.stats, ylog=False, view=False, filename=f'{filename}_stats.svg')
        visualise.plot_species(self.stats, view=False, filename=f'{filename}_species.svg')


    def run_simulation(self, genome = None):
        print("Running simulation...")
        self.frames = []
        self.done = False
        self.step = 0
        winner_net = neat.nn.FeedForwardNetwork.create(genome, self.config)
        obs = self.env.reset() 

        while not self.done and self.step < self.num_timesteps:

            if self.naive:
                print("No genome provided - falling back on random actions")
                action = self.env.action_space.sample()
            else:
                action = winner_net.activate(obs) 

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

    def write_winner_pkl(self):
        pass

    def save_simulation(self, filename='neat_sim.gif'):
        filename = f"{self.timestamp}_{filename}"
        imageio.mimsave(filename, self.frames, fps=30)
        print(f"Simulation saved to {filename}")


if __name__ == "__main__":
    idx = get_formatted_datetime()
    CONFIG_PATH = '/home/ubuntu/strahl-fs/project1/numpy-implmentation/slimevolleygym/slimevolleygym/xor_config'
    sim = Simulate(CONFIG_PATH, naive=False, num_timesteps=1000)
    genome = sim.generate_neat_network()
    sim.plot_genome(genome, f"neat_experiment_{idx}")
    sim.run_simulation(genome)
    sim.save_simulation()
