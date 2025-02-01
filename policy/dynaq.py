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
        self.model = np.zeros((len(self.action_space), self.height, self.width, 3))
        self.visited = np.zeros((len(self.action_space), self.height, self.width), dtype=bool)

    def reset(self):
        self.Q = np.zeros((len(self.action_space), self.height, self.width))
        self.model = np.zeros((len(self.action_space), self.height, self.width, 3))
        self.visited = np.zeros((len(self.action_space), self.height, self.width), dtype=bool)

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

        self.visited[action, state[0], state[1]] = True
        self.model[action, state[0], state[1]] = np.array([reward, next_state[0], next_state[1]])

        if not done:
            for i in range(self.n_planning):
                visited_actions, visited_y, visited_x = np.where(self.visited)
                idx = np.random.randint(len(visited_actions))
                s_action, s_y, s_x = visited_actions[idx], visited_y[idx], visited_x[idx]
                s_reward, s_next_y, s_next_x = self.model[s_action, s_y, s_x]
                max_Q = max(self.Q[:, int(s_next_y), int(s_next_x)])
                td_error = s_reward + self.gamma * max_Q - self.Q[s_action, s_y, s_x]
                self.Q[s_action, s_y, s_x] += self.alpha * td_error
