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
import os
import random
from datetime import datetime

import env_files.utils as utils
from env_files.action_data import ActionData
from env_files.safeplace_env import SafePlaceEnv
from env_files.state_data import StateData


class DatasetGeneration(SafePlaceEnv):
    def __init__(self):
        pass

    @staticmethod
    def generate_trajectories(iterations: int, real_env: bool, seeds: tuple, save_file=True):
        if len(seeds) != iterations:
            utils.error('(generate_trajectories) `seeds` length is different from `iterations`.')

        string = utils.dataset_header
        for iteration in range(iterations):
            random.seed(seeds[iteration])
            reservations_string, _ = utils.generate_reservations_file()
            utils.update_reservations(string=reservations_string, random_initial_temp_in=True)

            initial_state = utils.initial_state()
            s: StateData = initial_state

            while not s.is_terminal:
                a = random.choice(list(ActionData))
                next_s = DatasetGeneration.get_next_state(s, a, real_env=real_env)

                csv_row = DatasetGeneration.extract_transition_data(s, a, next_s)
                string += csv_row

                s = next_s

        timestamp = datetime.utcnow().strftime('%y%m%d_%H%M%S')
        if real_env:
            folder = 'datasets/real_transition_model'
            filename = folder + '/rtm_dataset_' + timestamp + '.csv'
        else:
            folder = 'datasets/simple_transition_model'
            filename = folder + '/stm_dataset_' + timestamp + '.csv'

        if save_file:
            if not os.path.exists(folder):
                os.makedirs(folder)

            with open(filename, 'a') as f:
                f.write(string)

        return filename

    @staticmethod
    def extract_transition_data(s: StateData, a: ActionData, next_s: StateData) -> str:
        window_open = 1 if a.is_window_open else 0
        sanitizer_active = 1 if a.is_sanitizer_active else 0

        return '%d,%f,%f,%f,%f,%d,%f,%d,%f,%f,%f\n' % (s.people,
                                                       s.co2,
                                                       s.voc,
                                                       s.temp_in,
                                                       s.temp_out,
                                                       window_open,
                                                       a.ach,
                                                       sanitizer_active,
                                                       next_s.co2,
                                                       next_s.voc,
                                                       next_s.temp_in
                                                       )

    @staticmethod
    def convert_state_action_pair(s: StateData, a: ActionData) -> tuple[tuple, tuple, list]:
        window_open = 1 if a.is_window_open else 0
        sanitizer_active = 1 if a.is_sanitizer_active else 0

        return (s.people, s.co2, s.voc, s.temp_in, s.temp_out), (window_open, a.ach, sanitizer_active), \
               [s.people, s.co2, s.voc, s.temp_in, s.temp_out, window_open, a.ach, sanitizer_active]

    @staticmethod
    def get_next_state(s: StateData, a: ActionData, real_env: bool) -> StateData:
        if real_env is True:
            return SafePlaceEnv.transition_model(s, a)
        else:
            return DatasetGeneration.simple_transition_model(s, a)

    @staticmethod
    def simple_transition_model(s: StateData, a: ActionData) -> StateData:
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
        next_co2 = DatasetGeneration.next_co2(co2, people, a)

        # get next_voc
        next_voc = DatasetGeneration.next_voc(voc, people, a)

        # get next_temp_in
        next_temp_in = DatasetGeneration.next_temp_in(temp_in, temp_out, a)

        # get next_s
        next_s = StateData(next_hour, next_minute, next_people, next_co2, next_voc, next_temp_in, next_temp_out)

        return next_s

    @staticmethod
    def simulate(s: StateData, a: ActionData) -> (StateData, float):
        next_s = DatasetGeneration.simple_transition_model(s, a)

        reward, _, _, _ = SafePlaceEnv.get_reward(s, a, next_s)

        return next_s, reward

    @staticmethod
    def get_prediction_error(s: StateData, a: ActionData) -> list[float, float, float]:
        real_next_s = SafePlaceEnv.transition_model(s, a)

        real_co2, real_voc, real_temp = real_next_s.co2, real_next_s.voc, real_next_s.temp_in

        other_next_s = DatasetGeneration.simple_transition_model(s, a)
        other_co2, other_voc, other_temp = other_next_s.co2, other_next_s.voc, other_next_s.temp_in

        errors = [abs(other_co2 - real_co2), abs(other_voc - real_voc), abs(other_temp - real_temp)]

        if utils.execnet:
            if utils.verbose:
                print('Prediction errors - absolute values - [`co2`, `voc`, `temp_in`]: ' + str(errors))

        return errors

    @staticmethod
    def next_co2(co2, people, action):
        if action.is_stack_ventilation_compatible:
            if people > 0:
                # devINFO: edited
                next_co2 = co2 + people * utils.time_delta / (action.ach * 150)
                # next_co2 = co2 + people * utils.time_delta / (action.ach * 5)
            else:
                if action.is_ventilation_active:
                    # devINFO: edited
                    next_co2 = co2 - action.ach * 20000 / utils.room_volume
                    # next_co2 = co2 - action.ach * 20
                else:
                    next_co2 = co2 - 100
        else:
            # devINFO: edited
            next_co2 = co2 - action.ach * 30000 / utils.room_volume
            # next_co2 = co2 - action.ach * 50

        if next_co2 < utils.outdoor_co2:
            next_co2 = utils.outdoor_co2

        return next_co2

    @staticmethod
    def next_voc(voc, people, action):
        if action.is_sanitizer_active and action.is_ventilation_active:
            voc_removed = utils.voc_removal_rate * action.ach / 7 * 2
        elif not action.is_sanitizer_active and action.is_ventilation_active:
            voc_removed = utils.voc_removal_rate * action.ach / 7
        elif action.is_sanitizer_active and not action.is_ventilation_active:
            voc_removed = utils.voc_removal_rate
        else:
            voc_removed = 0

        if people > 0:
            # devINFO: edited
            voc_produced = 100 * people
            # voc_produced = utils.voc_produced_per_person * people
        else:
            voc_produced = 0

        voc_delta = voc_produced - voc_removed
        voc_delta_concentration = voc_delta / utils.room_volume

        next_voc = voc + voc_delta_concentration

        if next_voc < utils.outdoor_voc:
            next_voc = utils.outdoor_voc

        return next_voc

    @staticmethod
    def next_temp_in(temp_in, temp_out, action):
        if action.is_window_open:
            delta = temp_out - temp_in
            sign = math.copysign(1, delta)
            delta = abs(delta)

            # devINFO: edited
            new_delta = delta / 1.5
            # new_delta = delta / 10

            new_delta = math.copysign(new_delta, sign)
            next_temp_in = temp_in + new_delta
        else:
            next_temp_in = temp_in

        return next_temp_in
