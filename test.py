from env import Environment
from simulator import Simulator

if __name__ == '__main__':
    sim = 1
    epi = 100
    env = Environment()
    s = Simulator(sim, epi, env)
    s.run()
