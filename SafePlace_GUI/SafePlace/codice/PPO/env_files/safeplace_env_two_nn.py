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

import pickle

from matrix_pypy import NNPyPy
from safeplace_env import SafePlaceEnv
from dataset_generation import DatasetGeneration
from state_data import StateData
from action_data import ActionData
import utils


class SafePlaceEnvTwoNN(SafePlaceEnv):
    def __init__(self, expert_weights_pickle_path: str):

        with open(expert_weights_pickle_path, 'rb') as f:
            nn_pypy = pickle.load(f)

        self.expert_nn: NNPyPy = nn_pypy
        self.real_nn = None
        self.nn = None

        self.observations = {}
        for action in ActionData:
            window_open = 1 if action.is_window_open else 0
            ach = action.ach
            sanitizer_active = 1 if action.is_sanitizer_active else 0
            key = (window_open, ach, sanitizer_active)

            self.observations[key] = []

        self.hits = 0
        self.miss = 0

        self.prediction = False
        self.last_pair_seen = None

        if utils.verbose:
            print('(SafePlaceEnvTwoNN) Environment initialized.')

    def update_real_nn(self, real_weights_pickle_path: str):
        with open(real_weights_pickle_path, 'rb') as f:
            nn_pypy = pickle.load(f)

        self.real_nn: NNPyPy = nn_pypy

        if utils.verbose:
            print('(SafePlaceEnvTwoNN) Real neural network weights updated.')

    def choose_nn(self, state: tuple, action: tuple):
        if self.real_nn is None:
            self.nn: NNPyPy = self.expert_nn
            return

        for state_observed in self.observations[action]:
            if utils.states_are_close(state, state_observed):
                self.hits += 1
                self.nn = self.real_nn
                return

        self.miss += 1
        self.nn = self.expert_nn

    def clear_stats(self):
        self.hits = 0
        self.miss = 0

    def simulate(self, s: StateData, a: ActionData) -> (StateData, float):
        # get next_hour and next_minute
        next_hour, next_minute = SafePlaceEnv.next_time(s.hour, s.minute)

        # get next_people and next_temp_out
        next_people = utils.reservations[next_hour][next_minute].people
        next_temp_out = utils.reservations[next_hour][next_minute].temp_out

        state, action, x = DatasetGeneration.convert_state_action_pair(s, a)
        if self.prediction:
            self.last_pair_seen = (state, action)

        self.choose_nn(state, action)
        y = self.nn.model_inference(x)

        next_co2 = y[0]
        next_voc = y[1]
        next_temp_in = y[2]

        # get next_s
        next_s = StateData(next_hour, next_minute, next_people, next_co2, next_voc, next_temp_in, next_temp_out)

        # get reward
        reward, _, _, _ = SafePlaceEnv.get_reward(s, a, next_s)

        return next_s, reward

    def get_prediction_error(self, s: StateData, a: ActionData) -> list[float, float, float]:
        real_next_s = SafePlaceEnv.transition_model(s, a)

        real_co2, real_voc, real_temp = real_next_s.co2, real_next_s.voc, real_next_s.temp_in

        self.prediction = True
        other_next_s, _ = self.simulate(s, a)
        self.prediction = False

        other_co2, other_voc, other_temp = other_next_s.co2, other_next_s.voc, other_next_s.temp_in

        errors = [abs(other_co2 - real_co2), abs(other_voc - real_voc), abs(other_temp - real_temp)]

        if utils.execnet:
            if utils.verbose:
                print('Prediction errors - absolute values - [`co2`, `voc`, `temp_in`]: ' + str(errors))
                print('Real nn: %s - Expert nn: %d' % (self.hits, self.miss))

        # Add state-action pair to `observations` dict
        self.observations[self.last_pair_seen[1]].insert(0, self.last_pair_seen[0])

        self.clear_stats()
        # IDEA: sometimes check if there are duplicates in dictionary's lists

        return errors
