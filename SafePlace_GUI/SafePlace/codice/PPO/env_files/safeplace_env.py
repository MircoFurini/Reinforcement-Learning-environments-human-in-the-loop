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

from env_files.environment import Environment
from env_files.state_data import StateData
from env_files.action_data import ActionData
import env_files.utils as utils


class SafePlaceEnv(Environment):
    def __init__(self):
        print('(SafePlaceEnv) The oracle is being utilized.')
        if utils.verbose:
            print('(SafePlaceEnv) Environment initialized.')

    @staticmethod
    def transition_model(s: StateData, a: ActionData) -> StateData:
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
    def simulate(s: StateData, a: ActionData) -> (StateData, float):
        """
        Simulator of the environment
            Input:
                - s: state at time t
                - a: action at time t
            Output:
                - next_s: state at time t + 1
                - reward: reward for 's x a -> next_s'
        """

        next_s = SafePlaceEnv.transition_model(s, a)

        # get reward
        reward, _, _, _ = SafePlaceEnv.get_reward(s, a, next_s)

        return next_s, reward

    @staticmethod
    def do_transition(s: StateData, a: ActionData) -> (StateData, tuple[float, float, float, float]):
        """
        Real environment
            Input:
                - s: state at time t
                - a: action at time t
            Output:
                - next_s: state at time t + 1
                - rewards: reward for 's x a -> next_s' and components that make up the main reward
        """

        next_s = SafePlaceEnv.transition_model(s, a)

        # get rewards
        rewards = SafePlaceEnv.get_reward(s, a, next_s)

        return next_s, rewards

    @staticmethod
    def get_prediction_error(s: StateData, a: ActionData) -> list[float, float, float]:
        return [0, 0, 0]

    @staticmethod
    def next_co2(co2, people, action):
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
                        # when 'window_ach' is 8 h^(-1), co2 decreases by 38.73% in an hour
                        # when 'vent_low_power_ach' is 9 h^(-1), co2 decreases by 42.45% in an hour
                        next_co2 = co2 * factor
                        # for 'next_co2' to get lower than 'CONST.outdoor_co2' you would need an ach greater than 40
                        # (but 'action.ach' is always between 0.1 and 9.97), so there is no need to verify that.
                    else:
                        next_co2 = co2 - (1 / (20 * utils.low_co2_descent_delta)) * delta * delta
                        # again, this equation will never lower 'next_co2' so that it goes below 'CONST.outdoor_co2'.

                    potential_variation = factor - 1
                else:
                    # co2 concentration reaches equilibrium
                    next_co2 = co2
                    potential_variation = 0
        else:
            if action.ach >= 15:
                # This calculation depends on the assumption 'CONST.max_people' = 50
                new_ach = action.ach * 50 / (10 / 49 * people + 39.8)
            else:
                # This calculation depends on the assumption 'CONST.max_people' = 50
                min_new_ach = 14.9  # with 50 people
                max_new_ach = 0.3666 * action.ach + 13.34499  # with 0 people
                m, q = utils.find_line_equation(0, max_new_ach, 50, min_new_ach)
                new_ach = m * people + q

            factor = math.exp(-(new_ach - 15.2) / 35)
            next_co2 = co2 * factor

            potential_variation = factor - 1

            if next_co2 < utils.outdoor_co2:
                next_co2 = utils.outdoor_co2

        return next_co2, potential_variation

    @staticmethod
    def next_voc(voc, potential_variation, people, action):
        if action.is_sanitizer_active:
            voc_removed = utils.voc_removal_rate
        else:
            voc_removed = 0

        if people > 0:
            voc_produced = utils.voc_produced_per_person * people
        else:
            voc_produced = 0

        voc_delta = voc_produced - voc_removed
        # NOTE: we assume that the voc produced gets immediately mixed with air
        voc_delta_concentration = voc_delta / utils.room_volume

        next_voc = voc + voc_delta_concentration

        if potential_variation < 0:
            next_voc = next_voc * (1 + potential_variation)

        if next_voc < utils.outdoor_voc:
            next_voc = utils.outdoor_voc

        return next_voc

    @staticmethod
    def next_temp_in(temp_in, temp_out, people, action):
        # devINFO: new update
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

            """if delta < 1.64872:
                new_delta = (1 / 2.50363) * delta * delta
            elif delta > 2.51188:
                new_delta = 2
            else:
                new_delta = 5 * math.log(delta, 10)"""

            if delta < 1.6:
                new_delta = 0.4 * delta * delta
            elif delta > 2.5:
                new_delta = 2
            else:
                new_delta = delta * utils.temp_m + utils.temp_q

            new_delta = math.copysign(new_delta, sign)

            new_delta += (temp_increase / 4)

            next_temp_in = temp_in + new_delta
        else:
            next_temp_in = temp_in + temp_increase

        return next_temp_in

    @staticmethod
    def get_reward(s: StateData, a: ActionData, next_s: StateData) -> tuple[float, float, float, float]:
        """
        - air_quality (co2, voc)
        - comfort (temp, noise)
        - energy_consumption (energy)
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
            # devINFO: new update
            temp_reward = math.exp(-(((temp_in - 20) / 5) * ((temp_in - 20) / 5)))
            # temp_reward = math.exp(-(((temp_in - 20) / 2) * ((temp_in - 20) / 2)))

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
    def next_time(hour, minute):
        minutes = utils.reservations[hour].keys()

        # get next_hour and next_minute
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
