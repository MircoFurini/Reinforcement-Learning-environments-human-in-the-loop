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
from datetime import datetime
import os
import csv

import env_files.utils as utils
from env_files.state_data import StateData


def create_stats_path_and_filename():
    p = utils.model_param

    time = datetime.utcnow().strftime('%y%m%d_%H%M%S')

    common = 'expconst_' + str(p.exp_const)
    common += '_iterations_' + str(p.iterations)
    common += '_maxdepth_' + str(p.max_depth)
    common += '_rolloutmoves_' + str(p.rollout_moves)

    stats_path = 'safeplace/' + common + '/'
    stats_filename = 'safeplace_' + time + '_' + common + '.csv'

    return stats_path, stats_filename, time


def step_stats_start():
    s_stats_time = time()
    return s_stats_time


def step_stats_record_data(s: StateData, action, rewards, s_stats_time, iteration_number, times_bn):
    s_stats_time = time() - s_stats_time

    sim_done = iteration_number + 1
    sim_wasted = times_bn + 1

    folder = utils.step_stats_folder + utils.stats_path
    filename = utils.stats_filename

    hour = s.hour
    minute = s.minute
    people = s.people
    co2 = s.co2
    voc = s.voc
    temp_in = s.temp_in
    temp_out = s.temp_out

    action = action.code

    date = datetime(1990, 1, 1, hour=hour, minute=minute)
    date = date.strftime('%H:%M')

    reward = rewards[0]
    air_quality_reward = rewards[1]
    comfort_reward = rewards[2]
    energy_reward = rewards[3]

    stats_row = [date, people, co2, voc, temp_in, temp_out, action, reward,
                 air_quality_reward, comfort_reward, energy_reward, s_stats_time, sim_done, sim_wasted]
    header = ['time', 'people', 'co2', 'voc', 'temp_in', 'temp_out', 'action', 'reward',
              'air_quality_reward', 'comfort_reward', 'energy_reward', 's_stats_time', 'sim_done', 'sim_wasted']

    update_csv(stats_row, header, folder, filename)


def make_csv(table, header, header_format, folder, filename):
    """
    Deprecated
    """
    # Creating folder if it doesn't already exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Creating csv file
    filename = folder + filename
    np.savetxt(filename, table, fmt=header_format, delimiter=",", header=header, comments='')


def update_csv(stats_row, header, folder, filename):
    def _update(new=False):
        with open(fullname, mode='a') as csv_file:
            file_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if new:
                file_writer.writerow(header)
            file_writer.writerow(stats_row)

    fullname = folder + filename
    try:
        with open(fullname, mode='r') as f:
            pass
        _update()
    # if file doesn't exist, create it
    except FileNotFoundError:
        if utils.verbose:
            print('File does not exist, creating it now.')
        if not os.path.exists(folder):
            os.makedirs(folder)

        _update(new=True)


def list_as_array_to_csv(list_as_array: list, folder, filename):
    fullname = folder + filename
    with open(fullname, mode='a') as csv_file:
        file_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in list_as_array:
            file_writer.writerow(row)

    return fullname


def read_csv_as_list(filepath: str):
    with open(filepath, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    return data


def format_reservations_history(reservations: list):
    for i, row in enumerate(reservations):
        for j, string in enumerate(row):
            new_cell = string.split('-')
            reservations_path = new_cell[0]
            initial_temp_in = float(new_cell[1])
            new_cell = (reservations_path, initial_temp_in)
            reservations[i][j] = new_cell

    return reservations


def load_reservations_history(filepath: str):
    data: list = read_csv_as_list(filepath)
    return format_reservations_history(data)
