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
from enum import Enum
from typing import Union

from utils import maximum_ach_stack_ventilation_compatible, no_ventilation_ach, window_ach, vent_low_power_ach, \
    vent_high_power_ach


class ActionData(Enum):
    """
    Enumeration of available actions in `SafePlaceEnv`.
    For further details about the attributes of the actions, look at the `__init__` method.
    """
    # action = (code, ach, noise_reward, energy_reward, sanitizer_active)
    ALL_OFF = (0, no_ventilation_ach, 1, 1, False)

    WINDOW_ON = (1, window_ach, 1, 1, False)
    VENT_LOW_ON = (2, vent_low_power_ach, 0.7, 0.8, False)
    VENT_HIGH_ON = (3, vent_high_power_ach, 0.1, 0.2, False)
    SANITIZER_ON = (4, no_ventilation_ach, 0.9, 0.8, True)

    WINDOW_SANITIZER_ON = (5, window_ach, 0.9, 0.8, True)
    VENT_LOW_SANITIZER_ON = (6, vent_low_power_ach, 0.6, 0.6, True)
    VENT_HIGH_SANITIZER_ON = (7, vent_high_power_ach, 0, 0, True)

    def __init__(self, code, ach, noise_reward, energy_reward, sanitizer_active) -> None:
        """
        Method called after the creation of the object. It just saves the data.

        Args:
            code: unique code that identifies the action.
            ach: air changes per hour (h^(-1)).
            noise_reward: reward related to the noise pollution of the action. This variable can assume values in the
                range [0, 1].
            energy_reward: reward related to the energy consumption of the action. This variable can assume values in
                the range [0, 1].
            sanitizer_active: boolean value.
        """
        self.code_number = code
        self.air_change_per_hour = ach
        self.noise = noise_reward
        self.energy = energy_reward
        self.sanitizer_active = sanitizer_active

    @staticmethod
    def get_action(code) -> Union['ActionData', None]:
        """
        Returns the action associated with the unique code.

        Args:
            code: the unique code.

        Returns: the `ActionData` object; if the code is wrong, it returns None.
        """
        for action in ActionData:
            if action.code == code:
                return action
        return None

    @property
    def code(self) -> int:
        """
        Property method to access the `code` attribute.

        Returns: the `code` attribute.
        """
        return self.code_number

    @property
    def ach(self) -> float:
        """
        Property method to access the `ach` attribute.

        Returns: the `ach` attribute.
        """
        return self.air_change_per_hour

    @property
    def noise_reward(self) -> float:
        """
        Property method to access the `noise_reward` attribute.

        Returns: the `noise_reward` attribute.
        """
        return self.noise

    @property
    def energy_reward(self) -> float:
        """
        Property method to access the `energy_reward` attribute.

        Returns: the `energy_reward` attribute.
        """
        return self.energy

    @property
    def is_stack_ventilation_compatible(self) -> bool:
        """
        Property method that checks if the action is stack ventilation compatible.

        Returns: True if stack ventilation compatible; otherwise, False.
        """
        return self.ach <= maximum_ach_stack_ventilation_compatible

    @property
    def is_ventilation_active(self) -> bool:
        """
        Property method that checks if the action has any kind of ventilation active (i.e., ach different from 0.1).

        Returns: True if any kind of ventilation is active; otherwise, False.
        """
        return self.ach > no_ventilation_ach

    @property
    def is_window_open(self) -> bool:
        """
        Property method that checks if the action opens the window (i.e., ach equal to `utils.window_ach`).

        Returns: True if the action opens the window; otherwise, False.
        """
        return self.ach == window_ach

    @property
    def is_sanitizer_active(self) -> bool:
        """
        Property method that checks if the action activates the sanitizer.

        Returns: True if the action activates the sanitizer; otherwise, False.
        """
        return self.sanitizer_active

    @property
    def k(self) -> float:
        """
        Property method that calculates `k` value of Equation (1) of the supplementary material.
        Follows Teleszewski and Gładyszewska-Fiedoruk 2018 paper.

        Returns: the computed `k` value associated with the action.
        """
        a = 1.35
        b = -1.261
        c = 0.945
        d = -0.236
        e = 0.005
        m = self.ach
        return a + b * m + c * math.pow(m, 1.5) + d * math.pow(m, 2) + e * math.pow(m, 3)
