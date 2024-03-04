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

from abc import abstractmethod


class Environment:
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def do_transition(self, *args, **kwargs):
        """
        This function does a transition given the current state and action to execute.
        Returns next state and reward.
        """
        pass

    @abstractmethod
    def get_reward(self, *args, **kwargs):
        pass

    @abstractmethod
    def simulate(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_prediction_error(self, *args, **kwargs):
        """
        Returns prediction absolute error of the current environment compared to the oracle (SafePlaceEnv).
        """
        pass
