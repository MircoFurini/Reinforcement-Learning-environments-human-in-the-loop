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

import random
import pickle

import utils
import stats
from model_parameters import ModelParams
from safeplace_env_two_nn import SafePlaceEnvTwoNN
from state_data import StateData
from state_node import StateNode
from safeplace_env import SafePlaceEnv
from dataset_generation import DatasetGeneration
from safeplace_env_nn_pypy import SafePlaceEnvNNPyPy


def uct(initial_state: StateData):
    root = StateNode(initial_state)
    new_data = []

    cumulative_reward = 0
    co2_abs_errors = []
    voc_abs_errors = []
    temp_in_abs_errors = []

    while not root.state.is_terminal:
        if utils.step_stats:
            s_stats_time = stats.step_stats_start()

        for iteration_number in range(utils.model_param.iterations):
            root.simulate_from_state()

        best_an = max(root.actions.values(), key=lambda an: an.q())
        a = best_an.action
        s = root.state
        real_s, rewards = utils.env.do_transition(s, a)
        errors = utils.env.get_prediction_error(s, a)
        new_data.append(DatasetGeneration.extract_transition_data(s, a, real_s))
        real_sn = StateNode(real_s)

        if utils.step_stats:
            stats.step_stats_record_data(s, a, rewards, s_stats_time, iteration_number, times_bn=0)

        if utils.batch_stats:
            cumulative_reward += rewards[0]
            co2_abs_errors.append(errors[0])
            voc_abs_errors.append(errors[1])
            temp_in_abs_errors.append(errors[2])

        # Since the best node and real node could have different values, a new tree must be created
        root = real_sn

    return new_data, cumulative_reward, co2_abs_errors, voc_abs_errors, temp_in_abs_errors


def oracle_uct(initial_state: StateData):
    new_env = SafePlaceEnv()
    old_env = utils.change_env(new_env)

    root = StateNode(initial_state)
    cumulative_reward = 0

    while not root.state.is_terminal:
        for iteration_number in range(utils.model_param.iterations):
            root.simulate_from_state()

        best_an = max(root.actions.values(), key=lambda an: an.q())
        a = best_an.action
        s = root.state
        real_s, rewards = utils.env.do_transition(s, a)
        real_sn = StateNode(real_s)

        cumulative_reward += rewards[0]

        # A new tree must be created in order to compare coherently with `uct` function
        root = real_sn

    utils.change_env(old_env)

    return cumulative_reward


def mcts(channel, _batches, _simulations, _datasets, _timestamp):
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

    if utils.verbose:
        print('Tree depth: %d\n' % mcts_param.max_depth)

    # Reproducibility ---
    if utils.reproducibility:
        random.seed(utils.temperature_seed)

        csv_files = {}

        for root, dirs, files in os.walk(utils.generated_reservations_folder):
            if len(dirs) == 0:
                room = root.split('/')[-1]
                for idx, file in enumerate(files):
                    files[idx] = root + '/' + file
                csv_files[room] = sorted(files)

        csv_files = dict(sorted(csv_files.items()))

        if not utils.all_rooms:
            reproducibility_reservations = csv_files[utils.room]
        else:
            reproducibility_reservations = []
            for key, item in csv_files.items():
                reproducibility_reservations += item

        utils.set_verbose(False)
        reproducibility_temp_in = {}
        for room, profile in csv_files.items():
            temps = []
            for reservations_filepath in profile:
                utils.update_reservations(reservations_filepath=reservations_filepath, random_initial_temp_in=True)
                temps.append(utils.initial_temp_in)
            reproducibility_temp_in[room] = temps
        utils.set_verbose(True)

        if not utils.all_rooms:
            reproducibility_temp_in = reproducibility_temp_in[utils.room]
        else:
            reproducibility_temp_in_temp = []
            for key, item in reproducibility_temp_in.items():
                reproducibility_temp_in_temp += item
            reproducibility_temp_in = reproducibility_temp_in_temp

        if len(reproducibility_temp_in) != (_batches * _simulations):
            utils.error('(mcts) Reproducibility error! Length of profile for room %s is not compatible with execnet '
                        'preferences.' % utils.room)

        simulation_idx = 0
    # ---

    fd_path = _datasets[0]
    vd_path = _datasets[1]

    batches_stats = []

    if utils.use_reservations_history:
        reservations = stats.load_reservations_history(utils.reservations_history_path)
        if len(reservations) != _batches:
            utils.error('(mcts) Reservations history csv file is not compatible with current `batches` preference.')
        if len(reservations[0]) != _simulations:
            utils.error('(mcts) Reservations history csv file is not compatible with current `simulations` preference.')
    else:
        reservations = []

    for batch in range(_batches):
        pickle_path = channel.receive()
        if utils.verbose:
            execnet_print('(mcts) Data received. Starting batch n.%d...' % (batch + 1), flush=True)
            execnet_print('(mcts) queue got: ' + pickle_path, flush=True)

        if not utils.use_two_neural_networks:
            if batch == 0:
                env = SafePlaceEnvNNPyPy(pickle_path)
            else:
                env.update_nn(pickle_path)
        else:
            if batch == 0:
                env = SafePlaceEnvTwoNN(pickle_path)
            else:
                env.update_real_nn(pickle_path)

        transition_data = [[], []]

        cr_list = []
        oracle_cr_list = []
        co2_errors_list = []
        voc_errors_list = []
        temp_in_errors_list = []

        reservations_list = []

        for simulation in range(_simulations):
            if utils.reproducibility:
                reservations_filepath = reproducibility_reservations[simulation_idx]
                utils.update_reservations(reservations_filepath=reservations_filepath)
                utils.set_initial_temp_in(reproducibility_temp_in[simulation_idx])
                simulation_idx += 1
            elif utils.use_reservations_history:
                reservations_tuple = reservations[batch][simulation]
                reservations_filepath = reservations_tuple[0]
                initial_temp_in = reservations_tuple[1]
                utils.update_reservations(reservations_filepath=reservations_filepath)
                utils.set_initial_temp_in(initial_temp_in)
            else:
                reservations_string, reservations_filepath = utils.generate_reservations_file(
                    save_file=utils.save_reservations_history)
                utils.update_reservations(string=reservations_string)

            utils.initialize_all(mcts_param, env)

            initial_state = utils.initial_state()
            new_data, cr, co2_ae, voc_ae, temp_in_ae = uct(initial_state)
            execnet_print(end='', flush=True)

            if utils.compare_with_oracle:
                oracle_cr = oracle_uct(initial_state)
                oracle_cr_list.append(oracle_cr)

            if utils.batch_stats:
                cr_list.append(cr)
                co2_errors_list += co2_ae
                voc_errors_list += voc_ae
                temp_in_errors_list += temp_in_ae

            if utils.save_reservations_history:
                reservations_data = reservations_filepath + '-' + '%.2f' % utils.initial_temp_in
                reservations_list.append(reservations_data)

            random.shuffle(new_data)
            index = int(len(new_data) * (1 - utils.dataset_factor))
            new_fd = new_data[:index]
            new_vd = new_data[index:]
            for row in new_fd:
                transition_data[0].append(row)
            for row in new_vd:
                transition_data[1].append(row)

            # The datasets always exist
            with open(fd_path, mode='a') as f:
                f.writelines(new_fd)
            with open(vd_path, mode='a') as f:
                f.writelines(new_vd)

        if utils.batch_stats:
            batch_stats = (cr_list, oracle_cr_list, co2_errors_list, voc_errors_list, temp_in_errors_list)
            batches_stats.append(batch_stats)

        if utils.save_reservations_history:
            reservations.append(reservations_list)

        channel.send(transition_data)
        if utils.verbose:
            execnet_print('(mcts) Data sent.', flush=True)

    if utils.batch_stats:
        folder = 'batch_stats/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = folder + 'safeplace_' + _timestamp + '.pickle'
        if utils.verbose:
            execnet_print('(mcts) Saving statistics in the following file: ' + filepath, flush=True)
        with open(filepath, 'wb') as f:
            pickle.dump(batches_stats, f)

    if utils.save_reservations_history:
        folder = 'batch_stats/'
        filename = 'reservations_history_' + _timestamp + '.csv'
        csv_path = stats.list_as_array_to_csv(reservations, folder, filename)
        if utils.verbose:
            execnet_print('(mcts) Saving reservations history in the following file: ' + csv_path, flush=True)

    if utils.verbose:
        execnet_print('(mcts) All %d batches completed! Exiting...' % _batches, flush=True)


if __name__ == '__channelexec__':
    import os
    os.dup2(2, 1)

    green = '\033[92m'
    endc = '\033[0m'

    def execnet_print(*args, **kwargs):
        new_args = []
        length = len(args)
        if length != 0:
            for idx, el in enumerate(args):
                if idx == 0 and length == 1:
                    new_args.append(green + args[0] + endc)
                elif idx == 0 and length > 1:
                    new_args.append(green + args[0])
                elif idx == length - 1:
                    new_args.append(args[idx] + endc)
                else:
                    new_args.append(args[idx])

            new_args = tuple(new_args)

            if 'sep' in kwargs and kwargs['sep'] != ' ':
                kwargs['sep'] = green + kwargs['sep'] + endc
            if 'end' in kwargs and kwargs['end'] != '\n':
                kwargs['end'] = green + kwargs['end'] + endc

            print(*new_args, **kwargs)
        else:
            print(*args, **kwargs)

    if utils.verbose:
        execnet_print('(mcts) Process initialized! Starting first batch of simulations...', flush=True)

    channel: globals()['channel'].__class__ = globals()['channel']
    batches, simulations, datasets, timestamp = channel.receive()
    mcts(channel, batches, simulations, datasets, timestamp)
