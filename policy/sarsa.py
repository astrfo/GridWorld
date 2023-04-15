import numpy as np

class Sarsa:
    def __init__(self):
        self.alpha = 0.01
        self.gamma = 0.9
        self.epsilon = 0.1
        self.action_space = np.array([0, 1, 2, 3])
        self.width = 6
        self.height = 4
        self.Q = np.zeros((len(self.action_space), self.height, self.width))

    def reset(self):
        self.Q = np.zeros((len(self.action_space), self.height, self.width))

    def action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            q_values = self.Q[:, state[0], state[1]]
            action = np.random.choice(np.where(q_values == max(q_values))[0])
        return action

    def update(self, state, action, reward, next_state, done):
        if done:
            next_action = None
            next_q = 0
        else:
            if np.random.rand() < self.epsilon:
                next_action = np.random.choice(self.action_space)
            else:
                q_values = self.Q[:, next_state[0], next_state[1]]
                next_action = np.random.choice(np.where(q_values == max(q_values))[0])
            next_q = self.Q[next_action, next_state[0], next_state[1]]
        td_error = reward + self.gamma * next_q - self.Q[action, state[0], state[1]]
        self.Q[action, state[0], state[1]] += self.alpha * td_error