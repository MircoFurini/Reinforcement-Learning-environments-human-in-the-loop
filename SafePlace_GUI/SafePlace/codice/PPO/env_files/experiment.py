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

import os
from time import time
import random

import utils
from dynamic_computation import uct
from model_parameters import ModelParams
from safeplace_env_nn_pypy import SafePlaceEnvNNPyPy

if __name__ == '__main__':
    if not utils.pypy:
        utils.error('(experiment) Set `pypy` to True.')

    exp_const = 10
    max_depth = 13
    rollout_moves = 0
    iterations_per_step = 10000

    # MCTS parameters
    mcts_param = ModelParams(
        exp_const=exp_const,
        max_depth=max_depth,
        rollout_moves=rollout_moves,
        iterations=iterations_per_step
    )

    csv_files = {}

    for root, dirs, files in os.walk('datasets/generated_reservations_profiles'):
        if len(dirs) == 0:
            room = root.split('/')[-1]
            for idx, file in enumerate(files):
                files[idx] = root + '/' + file
            csv_files[room] = sorted(files)

    csv_files = dict(sorted(csv_files.items()))

    temperature_seed = 1234
    random.seed(temperature_seed)

    env = SafePlaceEnvNNPyPy(
        'nn/pretrain/pickle/safeplace_simulated_nn_(15, 30, 40)_1000_220728_143059_seed_5_val_loss_0_00580.pickle')

    start = time()

    for room, profile in csv_files.items():
        for day, reservations_filepath in enumerate(profile):
            utils.initialize_all(mcts_param, env)
            if day == 1:
                break

            print('Room: %s - Day: %s' % (room, day))
            utils.update_reservations(reservations_filepath=reservations_filepath, random_initial_temp_in=True)

            initial_state = utils.initial_state()
            uct(initial_state)

    end = time()
    print('Time elapsed: ' + str(end - start))
