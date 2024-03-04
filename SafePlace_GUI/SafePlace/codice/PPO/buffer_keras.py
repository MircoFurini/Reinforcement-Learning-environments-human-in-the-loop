"""
Online Model Adaptation in Monte Carlo Tree Search Planning

This file is part of free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

It is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with the code.  If not, see <http://www.gnu.org/licenses/>.
"""

from copy import deepcopy

import numpy as np
from scipy.signal import lfilter

class Buffer:
    def __init__(self, capacity, obs_size):

        self.max_size = capacity
        self.idx = 0
        self.size = 0

        self.b_state = np.zeros((capacity, obs_size), dtype=np.float32)
        self.b_action = np.zeros(capacity, dtype=np.int32)
        self.b_logp = np.zeros(capacity, dtype=np.float32)
        self.b_reward = deepcopy(self.b_logp)
        self.b_vf = deepcopy(self.b_logp)

        self.b_adv_vf = deepcopy(self.b_logp)
        self.b_return = deepcopy(self.b_logp)

        self.gamma = 0.9
        self.lambd = 0.97

    def store(self, state, action, logp, reward, v):
        self.b_state[self.idx] = state
        self.b_action[self.idx] = action
        self.b_logp[self.idx] = logp

        self.b_reward[self.idx] = reward
        self.b_vf[self.idx] = v

        self.size = min(self.size+1, self.max_size)
        self.idx = (self.idx+1) % self.max_size

    def _discount_cumsum(self, x, discount):
        return lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def compute_mc(self, last_v, step):
        path_slice = slice(self.size - step, self.size)

        rewards = np.append(self.b_reward[path_slice], last_v)
        state_values = np.append(self.b_vf[path_slice], last_v)

        deltas = rewards[:-1] + self.gamma * state_values[1:] - state_values[:-1]
        self.b_adv_vf[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lambd)
        self.b_return[path_slice] = self._discount_cumsum(rewards, self.gamma)[:-1]

    def _normalize_adv(self):
        self.b_adv_vf[:self.size] = (self.b_adv_vf[:self.size] - np.mean(self.b_adv_vf[:self.size])) / (np.std(self.b_adv_vf[:self.size]) + 1e-10)

    def sample(self):
        self._normalize_adv()

        return self.b_state[:self.size], self.b_action[:self.size], self.b_logp[:self.size], self.b_adv_vf[:self.size], self.b_return[:self.size]

    def clear(self):
        self.idx = 0
        self.size = 0
