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

import math

from environment import Environment
from state_data import StateData
from action_data import ActionData


class SafePlaceEnv(Environment):
    """
    Subclass of `Environment`. The SafePlace environment representing the oracle.
    """
    def __init__(self) -> None:
        """
        Method called after the creation of the object. It just prints some info to terminal.
        """
        print('(SafePlaceEnv) The oracle is being utilized.')
        if utils.verbose:
            print('(SafePlaceEnv) Environment initialized.')

    @staticmethod
    def transition_model(s: StateData, a: ActionData) -> StateData:
        """
        Oracle's transition model. It predicts the next CO2 and VOCs concentrations and the next indoor temperature.
        In addition, uses the `utils.reservations` and `utils.referenced_reservations` dicts to get the next time and
        next forecasted outdoor temperature (that we assume to be the real one). This is a final method, i.e., the
        subclasses of `SafePlaceEnv` must not override this method.

        Args:
            s: `StateData` object representing the current state.
            a: `ActionData` object representing the action.

        Returns: the next `StateData` object representing the next state in the MDP.
        """

        hour = s.hour
        minute = s.minute
        people = s.people
        co2 = s.co2
        voc = s.voc
        temp_in = s.temp_in
        temp_out = s.temp_out

        # get next_hour and next_minute
        next_hour, next_minute = SafePlaceEnv.next_time(hour, minute)

        # get next_people and next_temp_out
        next_people = utils.reservations[next_hour][next_minute].people
        next_temp_out = utils.reservations[next_hour][next_minute].temp_out

        # get next_co2
        next_co2, potential_variation = SafePlaceEnv.next_co2(co2, people, a)

        # get next_voc
        next_voc = SafePlaceEnv.next_voc(voc, potential_variation, people, a)

        # get next_temp_in
        next_temp_in = SafePlaceEnv.next_temp_in(temp_in, temp_out, people, a)

        # get next_s
        next_s = StateData(next_hour, next_minute, next_people, next_co2, next_voc, next_temp_in, next_temp_out)

        return next_s


    @staticmethod
    def do_transition(s: StateData, a: ActionData) -> (StateData, tuple[float, float, float, float]):
        """
        Function used as interface for outside modules to obtain the next state and reward of the MDP transition using
        the oracle. Exclusively used outside MCTS simulation. This is a final method, i.e., the subclasses of
        `SafePlaceEnv` must not override this method.

        Args:
            s: `StateData` object representing the current state.
            a: `ActionData` object representing the action.

        Returns: i) the next `StateData` object representing the next state in the MDP and ii) the resulting reward and
        iii) its subcomponents.
        """
        next_s = SafePlaceEnv.transition_model(s, a)

        # get rewards
        rewards = SafePlaceEnv.get_reward(s, a, next_s)

        return next_s, rewards


    @staticmethod
    def next_co2(co2, people, action) -> (float, float):
        """
        Sub-model of the oracle's transition function that predicts the next indoor CO2 concentration.
        The computation depends on:
            - the current indoor CO2 concentration (ppm)
            - the current number of people inside the room (person)
            - the maximum occupancy of the room (person)
            - the action's ACH (air changes per hour) (h^(-1))
            - the time difference between the current CO2 concentration and the predicted one (min)
            - the room volume (m^3)
            - the outdoor CO2 concentration (ppm)
        Some values are directly taken from the `utils` module.

        The potential variation (`potential_variation` inside the function and also called relative variation in the
        paper) represents the variation of CO2 concentration before applying the correction of never going below the
        outdoor CO2 concentration. It is utilized by `SafePlaceEnv.next_voc()` to reduce the VOCs concentration ONLY
        when the potential variation is a negative number. In this case, it is always in the range (-1, 0):
            - line 166 (`potential_variation = (next_co2 - co2) / co2`): here `potential_variation` is always a positive
                number, so it will never be used by the sub-model responsible for VOCs prediction.
            - line 181 (`potential_variation = factor - 1`): it depends only on the calculation of `factor` at line 170
                (`factor = (1 - action.ach / 200)`) which is only executed when the action's ACH is in the range
                (0.1, 9.97]. In this range `factor` stays between 0 and 1 (open interval); therefore,
                also `potential_variation` stays in the range (-1, 0) because of line 181.
            - line 185 (`potential_variation = 0`): obvious. It will be ignored.
            - line 200 (second time that `potential_variation = factor - 1` appears): here `potential_variation` depends
                only on the calculation of `factor` at line 197 (`factor = math.exp(-(new_ach - 15.2) / 35)`). In the
                lines prior to 197, `new_ach` is always in the range [14.9, +infinity). Therefore, because of the
                exponential nature of line 197, `factor` will always be in the range (0, 1.01) with 1.01 rounded up.
                This in turn causes `potential_variation to be in the range (-1, 0.01)`. Even though the right extremity
                of the interval is greater than zero, this is not a problem. Recalling the beginning, when
                `potential_variation` is a positive number, it will be ignored by the sub-model responsible for VOCs
                prediction. For this reason, the new interval utilized by the sub-model is (-1, 0).

        Args:
            co2: the current indoor CO2 concentration.
            people: the current number of people inside the room.
            action: the `ActionData` object representing the action.

        Returns: i) the predicted next indoor CO2 concentration and ii) the potential variation.
        """
        if action.is_stack_ventilation_compatible:
            if people > 0:
                # following Teleszewski and Gładyszewska-Fiedoruk 2018 paper
                gamma = people / utils.room_volume
                next_co2 = utils.B * gamma * utils.time_delta * action.k + co2
                potential_variation = (next_co2 - co2) / co2
            else:
                delta = co2 - utils.outdoor_co2
                if action.is_ventilation_active:
                    factor = (1 - action.ach / 200)
                    if delta > utils.low_co2_descent_delta:
                        # When `window_ach` is 8 h^(-1), CO2 decreases by 38.73% in an hour.
                        # When `vent_low_power_ach` is 9 h^(-1), co2 decreases by 42.45% in an hour.
                        next_co2 = co2 * factor
                        # For `next_co2` to get lower than `utils.outdoor_co2` you would need an ach greater than 40
                        # (but `action.ach` is always between 0.1 and 9.97), so there is no need to verify that.
                    else:
                        next_co2 = co2 - (1 / (20 * utils.low_co2_descent_delta)) * delta * delta
                        # Again, this equation will never lower `next_co2` so that it goes below `utils.outdoor_co2`.

                    potential_variation = factor - 1
                else:
                    # CO2 concentration reaches equilibrium
                    next_co2 = co2
                    potential_variation = 0
        else:
            if action.ach >= 15:
                # This calculation depends on the assumption `utils.max_people` = 50
                new_ach = action.ach * 50 / (10 / 49 * people + 39.8)
            else:
                # This calculation depends on the assumption `utils.max_people` = 50
                min_new_ach = 14.9  # with 50 people
                max_new_ach = 0.3666 * action.ach + 13.34499  # with 0 people
                m, q = utils.find_linear_equation(0, max_new_ach, 50, min_new_ach)
                new_ach = m * people + q

            factor = math.exp(-(new_ach - 15.2) / 35)
            next_co2 = co2 * factor

            potential_variation = factor - 1

            if next_co2 < utils.outdoor_co2:
                next_co2 = utils.outdoor_co2

        return next_co2, potential_variation

    @staticmethod
    def next_voc(voc, potential_variation, people, action) -> float:
        """
        Sub-model of the oracle's transition function that predicts the next indoor VOCs concentration.
        The computation depends on:
            - the current indoor VOCs concentration (µg * m^(-3))
            - the current number of people inside the room (person)
            - the action's ACH (air changes per hour) (h^(-1))
            - the time difference between the current VOCs concentration and the predicted one (min) (implicit since in
                our experiments `utils.time_delta` has always been equal to 5)
            - the room volume (m^3)
            - the outdoor VOCs concentration (µg * m^(-3))
            - the sanitizer VOCs removal rate (µg) (again, the time is implicit, i.e., VOCs removed every 5 minutes)
        Some values are directly taken from the `utils` module.

        Args:
            voc: the current indoor VOCs concentration.
            potential_variation: potential variation of CO2 concentration (look at `SafePlaceEnv.next_co2()`
                description).
            people: the current number of people inside the room.
            action: the `ActionData` object representing the action.

        Returns: the predicted next indoor VOCs concentration.
        """
        if action.is_sanitizer_active:
            voc_removed = utils.voc_removal_rate
        else:
            voc_removed = 0

        if people > 0:
            voc_produced = utils.voc_produced_per_person * people
        else:
            voc_produced = 0

        voc_delta = voc_produced - voc_removed
        # We assume that the VOCs produced get immediately mixed with air
        voc_delta_concentration = voc_delta / utils.room_volume

        next_voc = voc + voc_delta_concentration

        if potential_variation < 0:
            next_voc = next_voc * (1 + potential_variation)

        if next_voc < utils.outdoor_voc:
            next_voc = utils.outdoor_voc

        return next_voc

    @staticmethod
    def next_temp_in(temp_in, temp_out, people, action) -> float:
        """
        Sub-model of the oracle's transition function that predicts the next indoor temperature.
        The computation depends on:
            - the current indoor temperature (°C)
            - the current outdoor temperature (°C)
            - the current number of people inside the room (person)
            - the action's ACH (air changes per hour) (h^(-1))
            - the time difference between the current indoor temperature and the predicted one (min) (implicit since in
                our experiments `utils.time_delta` has always been equal to 5)
            - the room volume (m^3)
            - the sanitizer VOCs removal rate (µg) (again, the time is implicit, i.e., VOCs removed every 5 minutes)
        Some values are directly taken from the `utils` module.

        Args:
            temp_in: the current indoor temperature.
            temp_out: the current outdoor temperature.
            people: the current number of people inside the room.
            action: the `ActionData` object representing the action.

        Returns: the predicted next indoor temperature.
        """
        people_increase = people * utils.time_delta / (utils.room_volume * 5)
        if action.is_sanitizer_active:
            sanitizer_increase = (utils.voc_removal_rate / 1000) * utils.time_delta / utils.room_volume
        else:
            sanitizer_increase = 0

        temp_increase = people_increase + sanitizer_increase

        if action.is_window_open:
            delta = temp_out - temp_in
            sign = math.copysign(1, delta)
            delta = abs(delta)

            if delta < 1.6:
                new_delta = 0.08 * delta * delta * utils.time_delta
            elif delta > 2.5:
                new_delta = 0.4 * utils.time_delta
            else:
                new_delta = (delta * utils.temp_m + utils.temp_q) / 5 * utils.time_delta

            new_delta = math.copysign(new_delta, sign)

            new_delta += (temp_increase / 4)

            next_temp_in = temp_in + new_delta
        else:
            next_temp_in = temp_in + temp_increase

        return next_temp_in

    @staticmethod
    def get_reward(s: StateData, a: ActionData, next_s: StateData) -> tuple[float, float, float, float]:
        """
        Given a complete transition tuple (current_state, action, next_state), returns the computed reward and its
        subcomponents. This is a final method, i.e., the subclasses of `SafePlaceEnv` must not override this method.

        The SafePlace environment has a reward that consists of three subcomponents:
            - air quality reward subcomponent: mean between CO2 reward and VOCs reward when there are persons in the
                room; otherwise, it is set to 0.
            - comfort reward subcomponent: weighted mean between temperature and noise rewards when there are
                persons in the room; otherwise, it is set to 0.
            - energy reward subcomponent: constant value associated with the action performed by the agent (see
                `action_data.py` or Table 2 of the supplementary material pdf).

        Args:
            s: `StateData` object representing the current state.
            a: `ActionData` object representing the action.
            next_s: `StateData` object representing the next state.

        Returns: i) the computed reward, ii) the air quality reward subcomponent, iii) the comfort reward subcomponent
        and iv) the energy reward subcomponent
        """

        people = s.people

        co2 = next_s.co2
        voc = next_s.voc
        temp_in = next_s.temp_in

        if people > 0:
            # air quality
            if co2 < utils.acceptable_co2:
                co2_reward = utils.co2_m * co2 + utils.co2_q
                if co2 < utils.outdoor_co2:
                    co2_reward = 1

            elif co2 < utils.ideal_max_co2:
                co2_reward = utils.co2_a * math.pow(co2, 2) + utils.co2_b * co2 + utils.co2_c
            else:
                co2_reward = 0

            if voc < utils.acceptable_voc:
                voc_reward = utils.voc_m * voc + utils.voc_q
                if voc < utils.outdoor_voc:
                    voc_reward = 1

            elif voc < utils.ideal_max_voc:
                voc_reward = utils.voc_a * math.pow(voc, 2) + utils.voc_b * voc + utils.voc_c
            else:
                voc_reward = 0

            air_quality_reward = ((co2_reward + voc_reward) / 2) * utils.air_quality_factor

            # comfort
            temp_reward = math.exp(-(((temp_in - 20) / 5) * ((temp_in - 20) / 5)))

            noise_reward = a.noise_reward

            comfort_reward = ((3 * temp_reward + noise_reward) / 4) * utils.comfort_factor
        else:
            air_quality_reward = 1
            comfort_reward = 1

        # energy consumption
        energy_reward = a.energy_reward * utils.energy_factor

        reward = (air_quality_reward + comfort_reward + energy_reward) / \
                 (utils.air_quality_factor + utils.comfort_factor + utils.energy_factor)

        return reward, air_quality_reward, comfort_reward, energy_reward

    @staticmethod
    def next_time(hour, minute) -> (int, int):
        """
        Uses the `utils.reservations` and `utils.referenced_reservations` dicts to get the next time and next forecasted
        outdoor temperature (that we assume to be the real one). This is a final method, i.e., the subclasses of
        `SafePlaceEnv` must not override this method.

        Args:
            hour: current hour of the state.
            minute: current minute of the state.

        Returns: i) the next hour and ii) the next minute.
        """
        minutes = utils.reservations[hour].keys()

        # Get next_hour and next_minute
        if max(minutes) == minute:
            index_of_hour = utils.referenced_reservations['hours']['ki'][hour]
            index_of_next_hour = index_of_hour + 1

            if index_of_next_hour in utils.referenced_reservations['hours']['ik']:
                next_hour = utils.referenced_reservations['hours']['ik'][index_of_next_hour]
            else:
                utils.error('(simulate) Error! Could not find next hour')

            next_minute = utils.referenced_reservations['minutes'][next_hour]['ik'][0]
        else:
            next_hour = hour

            index_of_minute = utils.referenced_reservations['minutes'][hour]['ki'][minute]
            index_of_next_minute = index_of_minute + 1

            if index_of_next_minute in utils.referenced_reservations['minutes'][hour]['ik']:
                next_minute = utils.referenced_reservations['minutes'][hour]['ik'][index_of_next_minute]
            else:
                utils.error('(simulate) Error! Could not find next minute')

        return next_hour, next_minute
