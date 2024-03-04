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

from env_files.utils import maximum_ach_stack_ventilation_compatible, no_ventilation_ach, window_ach, vent_low_power_ach, \
    vent_medium_power_ach, vent_high_power_ach


class ActionData(Enum):
    # action = (code, ach, noise_reward, energy_reward, sanitizer_active)
    ALL_OFF = (0, no_ventilation_ach, 1, 1, False)

    WINDOW_ON = (1, window_ach, 1, 1, False)
    VENT_LOW_ON = (2, vent_low_power_ach, 0.7, 0.8, False)
    VENT_HIGH_ON = (3, vent_high_power_ach, 0.1, 0.2, False)
    SANITIZER_ON = (4, no_ventilation_ach, 0.9, 0.8, True)

    WINDOW_SANITIZER_ON = (5, window_ach, 0.9, 0.8, True)
    VENT_LOW_SANITIZER_ON = (6, vent_low_power_ach, 0.6, 0.6, True)
    VENT_HIGH_SANITIZER_ON = (7, vent_high_power_ach, 0, 0, True)

    def __init__(self, code, ach, noise_reward, energy_reward, sanitizer_active):
        self.code_number = code
        self.air_change_per_hour = ach
        self.noise = noise_reward
        self.energy = energy_reward
        self.sanitizer_active = sanitizer_active

    @staticmethod
    def get_action(code):
        for action in ActionData:
            if action.code == code:
                return action
        return None

    @property
    def code(self):
        return self.code_number

    @property
    def ach(self):
        return self.air_change_per_hour

    @property
    def noise_reward(self):
        return self.noise

    @property
    def energy_reward(self):
        return self.energy

    @property
    def is_stack_ventilation_compatible(self):
        return self.ach <= maximum_ach_stack_ventilation_compatible

    @property
    def is_ventilation_active(self):
        return self.ach > no_ventilation_ach

    @property
    def is_window_open(self):
        return self.ach == window_ach

    @property
    def is_sanitizer_active(self):
        return self.sanitizer_active

    @property
    def k(self):
        # following Teleszewski and Gładyszewska-Fiedoruk 2018 paper
        a = 1.35
        b = -1.261
        c = 0.945
        d = -0.236
        e = 0.005
        m = self.ach
        return a + b * m + c * math.pow(m, 1.5) + d * math.pow(m, 2) + e * math.pow(m, 3)
