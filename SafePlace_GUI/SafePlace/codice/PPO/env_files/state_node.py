# -*- coding: utf-8 -*-
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

import math
import random
from collections import defaultdict

import utils
from action_node import ActionNode
from state_data import StateData
from action_data import ActionData


class StateNode:
    def __init__(self, s: StateData):
        self.state = s

        # Number of visits
        self.n = 0
        # The total value of the node
        self.v = 0

        # Child nodes (actions)
        self.actions = defaultdict(ActionNode)
        # Create all the action nodes
        self.expand_action_nodes()

    def expand_action_nodes(self):
        for a in ActionData:
            self.actions[a.code] = ActionNode(a)

    def select_action(self) -> int:
        actions_to_simulate = []
        for action_code, an in self.actions.items():
            if an.n == 0:
                actions_to_simulate.append(action_code)

        if len(actions_to_simulate) != 0:
            return random.choice(actions_to_simulate)
        else:
            ucb1_values = defaultdict(float)
            for action_code, an in self.actions.items():
                ucb1_values[action_code] = an.q() + utils.model_param.exp_const * \
                                           math.sqrt(math.log(self.n) / an.n)

            return max(ucb1_values, key=ucb1_values.get)

    def simulate_from_state(self, depth=0):
        action_code = self.select_action()

        reward = self.actions[action_code].simulate_from_action(self, depth=depth)

        self.n += 1
        self.v += reward

        return reward

    @staticmethod
    def rollout(s: StateData, depth, rollout_depth=0) -> float:
        # This line will have to be changed when dynamic actions is implemented (i.e. expand_action_nodes)
        a = random.choice(list(ActionData))

        if depth >= utils.model_param.max_depth or s.is_terminal or \
                rollout_depth >= utils.model_param.rollout_moves:
            reward = 0
        else:
            next_s, reward = utils.env.simulate(s, a)
            rollout_reward = StateNode.rollout(next_s, depth + 1, rollout_depth + 1)

            reward = reward + rollout_reward

        return reward
