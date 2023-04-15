import numpy as np

class Environment:
    def __init__(self):
        self.map = np.array(
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1]]
        )
        self.actions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        self.width = self.map.shape[1]
        self.height = self.map.shape[0]
        self.start_state = np.array([0, 0])
        self.current_state = self.start_state

    def reset(self):
        self.current_state = self.start_state
        return self.current_state

    def step(self, action):
        if action == 0 and self.current_state[0] > 0:
            self.current_state += self.actions[action]
        elif action == 1 and self.current_state[0] < (self.height-1):
            self.current_state += self.actions[action]
        elif action == 2 and self.current_state[1] > 0:
            self.current_state += self.actions[action]
        elif action == 3 and self.current_state[1] < (self.width-1):
            self.current_state += self.actions[action]
        
        reward = self.map[self.current_state[0], self.current_state[1]]
        if reward > 0:
            terminated, truncated, info = True, False, False
        else:
            terminated, truncated, info = False, False, False
        
        return self.current_state, reward, terminated, truncated, info
    
    def close(self):
        pass