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

# Class of global parameters
class ModelParams:
    def __init__(self, exp_const, max_depth, rollout_moves, iterations):
        info = ''
        # Exploration constant
        self.exp_const = exp_const
        # NO discount factor - because the env is finite horizon
        # self.gamma = gamma

        # Max depth reachable by simulating
        self.max_depth = max_depth
        if max_depth == 0:
            info += 'Since horizon is set to 0, there will be no limit on the depth of the tree.' + '\n'
            self.max_depth = float('+inf')

        # Max depth reachable by simulating set manually
        self.rollout_moves = rollout_moves
        if rollout_moves == 0:
            info += 'Since rollout_moves is set to 0, the rollout will go until it reaches the horizon limit or ' \
                    'a terminal state.' + '\n'
            self.rollout_moves = float('+inf')

        self.iterations = iterations

        print(info)
