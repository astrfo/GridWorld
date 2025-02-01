from tqdm import tqdm
from copy import deepcopy
from collector import Collector
from policy.qlearning import QLearning
from policy.sarsa import Sarsa
from policy.dynaq import DynaQ

class Simulator:
    def __init__(self, sim, epi, env):
        self.sim = sim
        self.epi = epi
        self.env = env
        self.policy = QLearning()
        self.collector = Collector(sim, epi)

    def run(self):
        for s in tqdm(range(self.sim)):
            self.policy.reset()
            self.collector.reset()
            for e in range(self.epi):
                state = self.env.reset()
                terminated, truncated, total_reward, total_step = False, False, 0, 0
                while not (terminated or truncated):
                    action = self.policy.action(state)
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                    self.policy.update(state, action, reward, next_state, (terminated or truncated))
                    state = deepcopy(next_state)
                    total_reward += reward
                    total_step += 1
                self.collector.collect_episode_data(total_reward, total_step)
            self.collector.save_episode_data()
        self.collector.save_simulation_data()
        self.env.close()
