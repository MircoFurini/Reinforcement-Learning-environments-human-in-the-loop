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
    """
    Abstract class used as interface for MCTS and neural network functions.
    """
    @abstractmethod
    def __init__(self, *args, **kwargs):
        """
        Every `Environment` subclass must implement this method. It can also do nothing.

        Args:
            *args: parameters of the subclass `__init__` method.
            **kwargs: additional parameters of the subclass `__init__` method in the form of a dict-like structure.
        """
        pass

    @abstractmethod
    def do_transition(self, *args, **kwargs):
        """
        This function does a transition using the oracle given the current state and action to execute.

        Args:
            *args: parameters of the subclass `do_transition` method.
            **kwargs: additional parameters of the subclass `do_transition` method in the form of a dict-like structure.

        Returns: i) the next state and ii) relative reward and iii) its subcomponents.
        """
        pass

    @abstractmethod
    def get_reward(self, *args, **kwargs):
        """
        Given a complete transition tuple (current_state, action, next_state), returns the computed reward and its
        subcomponents.

        Args:
            *args: parameters of the subclass `get_reward` method.
            **kwargs: additional parameters of the subclass `get_reward` method in the form of a dict-like structure.

        Returns: i) the computed reward and ii) its subcomponents.
        """
        pass

    @abstractmethod
    def simulate(self, *args, **kwargs):
        """
        Function used as interface for outside modules to obtain the next state and reward of the MDP transition given
        the current state and action to execute. Exclusively used in MCTS simulation.

        Args:
            *args: parameters of the subclass `simulate` method.
            **kwargs: additional parameters of the subclass `method` method in the form of a dict-like structure.

        Returns: i) the next state in the MDP and ii) the resulting reward.
        """
        pass

    @abstractmethod
    def get_prediction_error(self, *args, **kwargs):
        """
        Given the current state and the action, returns the predictions absolute errors of the current environment
        compared to the oracle (i.e., transition model of `SafePlaceEnv` class).

        Args:
            *args: parameters of the subclass `get_prediction_error` method.
            **kwargs: additional parameters of the subclass `get_prediction_error` method in the form of a dict-like
                structure.

        Returns: the predictions absolute errors.
        """
        pass
