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

from time import time

import utils
import stats
from model_parameters import ModelParams
from state_node import StateNode
from safeplace_env import SafePlaceEnv
from safeplace_env_nn_pypy import SafePlaceEnvNNPyPy

if not utils.pypy:
    from neural_network import NN
    from safeplace_env_nn_tf import SafePlaceEnvNNTF


def uct(initial_state):
    root = StateNode(initial_state)
    n_move = 0
    while not root.state.is_terminal:
        times_bs = 0
        best_s = root.state

        if utils.step_stats:
            s_stats_time = stats.step_stats_start()

        for iteration_number in range(utils.model_param.iterations):
            root.simulate_from_state()

            best_an = max(root.actions.values(), key=lambda an: an.q())
            actual_bs = best_an.state.state
            if not best_s == actual_bs:
                best_s = actual_bs
                times_bs = 0
            else:
                times_bs += 1

        a = best_an.action
        s = root.state
        real_s, rewards = utils.env.do_transition(s, a)
        errors = utils.env.get_prediction_error(s, a)
        real_sn = StateNode(real_s)

        if utils.step_stats:
            stats.step_stats_record_data(s, a, rewards, s_stats_time, iteration_number, times_bs)

        if utils.verbose:
            print('Best action: ' + str(a))
            print('Best state: ' + str(best_s))
            print('Real state: ' + str(real_s))
            print('Prediction errors: ' + str(errors) + '\n')

        # Since the best node and real node could have different values, a new tree must be created
        root = real_sn

        n_move += 1


if __name__ == '__main__':
    # Best parameters found so far (with 'utils.energy_factor' = 0.1)
    exp_const = 10
    # max_depth = ceil(log(0.01, gamma))
    max_depth = 13
    rollout_moves = 0
    # Set 'iterations_per_step' = 100'000 if better rewards are needed
    iterations_per_step = 10000

    utils.update_reservations(reservations_filepath=utils.reservation_profile_path)

    # MCTS parameters
    mcts_param = ModelParams(
        exp_const=exp_const,
        max_depth=max_depth,
        rollout_moves=rollout_moves,
        iterations=iterations_per_step
    )

    # Parameters of the environment
    if utils.adaptability_nn_pypy:
        env = SafePlaceEnvNNPyPy('nn/pickle/safeplace_simulated_nn_(15, 30, 40)_5000_220726_121543.pickle')
    elif utils.adaptability_nn_tf:
        nn = NN(model_path='...h5')
        env = SafePlaceEnvNNTF(nn)
    else:
        env = SafePlaceEnv()

    # Model initialization
    utils.initialize_all(mcts_param, env)

    print('Tree depth: %d\n' % mcts_param.max_depth)

    _initial_state = utils.initial_state()

    start = time()
    uct(_initial_state)
    end = time()

    print('Time elapsed: ' + str(end - start))
