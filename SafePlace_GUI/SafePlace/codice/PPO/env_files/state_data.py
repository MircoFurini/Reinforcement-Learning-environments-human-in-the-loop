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

import env_files.utils as utils


class StateData:
    def __init__(self, hour, minute, people, co2, voc, temp_in, temp_out):
        self.__s = [hour, minute, people, co2, voc, temp_in, temp_out]

    @property
    def is_terminal(self) -> bool:
        return utils.last_hour == self.hour and utils.last_minute == self.minute

    @property
    def s(self) -> list:
        return self.__s

    def __eq__(self, other):
        """
        if not isinstance(other, StateData):
            return NotImplemented
        """
        return self.s == other.s

    def __repr__(self):
        res = '%02d:%02d - people: %d - co2: %.2f - voc: %.2f - temp_in: %.2f - temp_out: %.2f' % \
              (self.hour, self.minute, self.people, self.co2, self.voc, self.temp_in, self.temp_out)

        return res

    @property
    def hour(self) -> int:
        return self.s[0]

    @property
    def minute(self) -> int:
        return self.s[1]

    @property
    def people(self) -> int:
        return self.s[2]

    @property
    def co2(self) -> float:
        return self.s[3]

    @co2.setter
    def co2(self, value):
        self.s[3] = value

    @property
    def voc(self) -> float:
        return self.s[4]

    @voc.setter
    def voc(self, value):
        self.s[4] = value

    @property
    def temp_in(self) -> float:
        return self.s[5]

    @temp_in.setter
    def temp_in(self, value):
        self.s[5] = value

    @property
    def temp_out(self) -> float:
        return self.s[6]
