import numpy as np

class DynaQ:
    def __init__(self):
        self.alpha = 0.01
        self.gamma = 0.9
        self.epsilon = 0.1
        self.n_planning = 10
        self.action_space = np.array([0, 1, 2, 3])
        self.width = 6
        self.height = 4
        self.Q = np.zeros((len(self.action_space), self.height, self.width))
        self.model = np.zeros((len(self.action_space), self.height, self.width))

    def reset(self):
        self.Q = np.zeros((len(self.action_space), self.height, self.width))
        self.model = np.zeros((len(self.action_space), self.height, self.width))

    def action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            q_values = self.Q[:, state[0], state[1]]
            action = np.random.choice(np.where(q_values == max(q_values))[0])
        return action

    def update(self, state, action, reward, next_state, done):
        max_Q = max(self.Q[:, next_state[0], next_state[1]])
        td_error = reward + self.gamma * max_Q  - self.Q[action, state[0], state[1]]
        self.Q[action, state[0], state[1]] += self.alpha * td_error
        self.model[action, state[0], state[1]] = [reward, next_state]
        for i in range(self.n_planning):
            s_width = np.random.randint(6)
            s_height = np.random.randint(4)
            a = np.random.choice(range(self.action_space))
            r, ns = self.model[a, s_height, s_width]
            max_Q = max(self.Q[:, ns[0], ns[1]])
            td_error = reward + self.gamma * max_Q  - self.Q[a, s_height, s_width]
            self.Q[a, s_height, s_width] += self.alpha * td_error