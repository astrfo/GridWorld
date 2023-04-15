from env import Environment
from simulator import Simulator

if __name__ == '__main__':
    sim = 100
    epi = 1000
    s = Simulator(sim, epi)
    s.run()