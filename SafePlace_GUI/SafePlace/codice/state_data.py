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


class StateData:
    """
    Class to wrap the state of the MDP.
    For further details about the attributes of the state, look at the `__init__` method.
    """

    def __init__(self, hour, minute, people, co2, voc, temp_in, temp_out) -> None:
        """
        Method called after the creation of the object. It just saves the data as a list.

        Args:
            hour: hour of the state.
            minute: minute of the state. Together with hour, they represent a unique way of identifying the state on
                the current day.
            people: number of people (person).
            co2: concentration of CO2 (ppm).
            voc: concentration of VOCs (volatile organic compounds) (µg * m^(-3)).
            temp_in: indoor temperature (°C).
            temp_out: outdoor temperature (°C).
        """
        self.__s = [hour, minute, people, co2, voc, temp_in, temp_out]

    @property
    def is_terminal(self) -> bool:
        """
        Property method that checks if the given state is terminal (in our experiments `utils.last_hour` = 18 and
        `utils.last_minute = 0`).

        Returns: True if the state is terminal; otherwise, False.
        """
        return utils.last_hour == self.hour and utils.last_minute == self.minute

    @property
    def s(self) -> list:
        """
        Property method to access the state list.

        Returns: the state list.
        """
        return self.__s

    def __eq__(self, other) -> bool:
        """
        Overrides standard `__eq__` method. It allows == comparison in boolean clauses.

        Args:
            other: other object.

        Returns: True if the other object's list is equal to the current one.
        """
        return self.s == other.s

    def __repr__(self) -> str:
        """
        Overrides standard `__repr__` method. It allows to directly print the object to terminal with
        `print(state_data_object)`.

        Returns: the string representing the object.
        """
        res = '%02d:%02d - people: %d - co2: %.2f - voc: %.2f - temp_in: %.2f - temp_out: %.2f' % \
              (self.hour, self.minute, self.people, self.co2, self.voc, self.temp_in, self.temp_out)

        return res

    @property
    def hour(self) -> int:
        """
        Property method to access the `hour` attribute.

        Returns: the `hour` attribute.
        """
        return self.s[0]

    @property
    def minute(self) -> int:
        """
        Property method to access the `minute` attribute.

        Returns: the `minute` attribute.
        """
        return self.s[1]

    @property
    def people(self) -> int:
        """
        Property method to access the `people` attribute.

        Returns: the `people` attribute.
        """
        return self.s[2]

    @property
    def co2(self) -> float:
        """
        Property method to access the `co2` attribute.

        Returns: the `co2` attribute.
        """
        return self.s[3]

    @property
    def voc(self) -> float:
        """
        Property method to access the `voc` attribute.

        Returns: the `voc` attribute.
        """
        return self.s[4]

    @property
    def temp_in(self) -> float:
        """
        Property method to access the `temp_in` attribute.

        Returns: the `temp_in` attribute.
        """
        return self.s[5]

    @property
    def temp_out(self) -> float:
        """
        Property method to access the `temp_out` attribute.

        Returns: the `temp_out` attribute.
        """
        return self.s[6]
