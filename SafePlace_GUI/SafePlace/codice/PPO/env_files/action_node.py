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

import utils
import state_node as s_n
from action_data import ActionData
from state_data import StateData


class ActionNode:
    """
    An action node, i.e. an instance representing an action in the tree
    (linked to a parent state node, so basically a state-action node)
    """

    def __init__(self, a: ActionData):
        self.action = a

        # Number of visits
        self.n = 0
        # The total value of the node
        self.v = 0

        # In this MDP doing an action creates only a single child state node
        self.state: s_n.StateNode = None
        self.state_reward: float = None

    # The value of action node
    def q(self):
        if self.n != 0:
            return self.v / self.n
        else:
            return 0

    # Simulates from the action (i.e. state-action node)
    def simulate_from_action(self, sn, depth):
        s: StateData = sn.state

        # Check if the max depth or a terminal state have been reached
        if depth >= utils.model_param.max_depth or s.is_terminal:
            reward = 0
        else:
            if self.state is None:
                next_s, reward = utils.env.simulate(s, self.action)
                next_sn = s_n.StateNode(next_s)
                self.state = next_sn
                self.state_reward = reward

                rollout_reward = s_n.StateNode.rollout(next_s, depth + 1)
                self.state.n += 1
                self.state.v += rollout_reward

                propagation_reward = rollout_reward

            elif utils.check_tree_integrity:
                next_s, _ = utils.env.simulate(s, self.action)
                if self.state.state == next_s:
                    propagation_reward = self.state.simulate_from_state(depth + 1)
                else:
                    utils.error('(simulate_from_action) Error! self.state.state: StateData != next_s')
            else:
                propagation_reward = self.state.simulate_from_state(depth + 1)

            reward = self.state_reward + propagation_reward

        self.n += 1
        self.v += reward

        return reward
