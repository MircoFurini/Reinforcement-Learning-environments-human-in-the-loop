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
import sys
import csv
import random
import math
from collections import defaultdict
from datetime import datetime

import stats


# Seeds ---
seed = 582039      # Python standard library seed
random.seed(seed)


# Experimental settings ---
temperature_seed = 1243


# General preferences ---
verbose = True                # print main info on terminal
step_stats = True             # save general statistics for every iteration/step on a csv file

# In `dynamic_computation.py`, it allows to utilize the reservations file specified here
reservation_profile_path = 'SafePlace/codice/datasets/reservations_test.csv'
# ---



# Environment settings ---
reservations_generation_first_hour = 8
reservations_generation_last_hour = 18
reservations_generation_min_temp = -5  # (°C)
reservations_generation_max_temp = 40  # (°C)

random_initial_temp_in_min = 17        # (°C)
random_initial_temp_in_max = 30        # (°C)
random_initial_temp_in_threshold = 5   # (°C)
random_initial_temp_out_max = 30       # (°C)

max_people = 50  # (person)
time_delta = 5   # (min) do not change this variable unless you've already modified the 'reservations_*.csv' file

B = 180                                          # (m^3 * ppm * person^(-1) * min^(-1)) (Teleszewski and Gładyszewska-Fiedoruk 2018)
maximum_ach_stack_ventilation_compatible = 9.97  # (h^(-1)) do not change this constant
voc_produced_per_person_original = 6250          # (µg * h^(-1) * person^(-1)) (Tang et al. 2016)
voc_produced_per_person = voc_produced_per_person_original / 60 * time_delta  # (µg * person^(-1))

voc_removal_rate = 10000     # (µg) voc removed by sanitizer per step

room_volume = 300            # (m^3)
initial_temp_in = 19.0       # (°C)
initial_ach = 0.1            # (h^(-1))
initial_co2 = 500            # (ppm)
initial_voc = 50             # (µg * m^(-3))

acceptable_co2 = 1000        # (ppm)
acceptable_voc = 600         # (µg * m^(-3))

outdoor_co2 = 400            # ppm
low_co2_descent_delta = 100  # ppm

outdoor_voc = 30             # (µg * m^(-3))

no_ventilation_ach = 0.1     # (h^(-1))
window_ach = 8               # (h^(-1))
vent_low_power_ach = 11      # (h^(-1))
vent_high_power_ach = 21     # (h^(-1))

# Reward constants
ideal_max_co2 = 2500         # ppm
ideal_max_voc = 1500         # (µg * m^(-3))

max_reward_co2 = 1
min_reward_co2 = 0.7
max_reward_voc = 1
min_reward_voc = 0.7


def find_linear_equation(x1, y1, x2, y2) -> (float, float):
    """
    Finds the unique linear equation given two points.

    Args:
        x1: x coordinate of first point.
        y1: x coordinate of first point.
        x2: x coordinate of first point.
        y2: x coordinate of first point.

    Returns: i) `m` and ii) `q` of the line.
    """
    __m = (y1 - y2) / (x1 - x2)
    __q = y1 - __m * x1
    # 'y = %.5f * x + %.5f' % (m, q)
    return __m, __q


def find_parabola_with_discriminant_zero(x_min_reward, min_reward, x_zero_reward) -> (float, float, float):
    """
    Finds a parabola with discriminant zero, i.e., its intersections with x-axis is only one. It receives the starting
    point and the x coordinate of the point that reaches 0, i.e., (x, 0).

    Args:
        x_min_reward: x coordinate of the starting point.
        min_reward: y coordinate of the starting point.
        x_zero_reward: x coordinate of the point that lies on x-axis.

    Returns: i) `a`, ii) `b`, iii) `c` considering a parabola of equation y(x) = `a` * x^2 + `b` * x + `c`.
    """
    _min = min_reward
    acceptable = x_min_reward  # beta
    ideal_max = x_zero_reward  # alpha

    __d = math.pow(ideal_max - acceptable, 2)

    __a = _min / __d
    __b = (-2) * ideal_max * _min / __d
    __c = math.pow(ideal_max, 2) * _min / __d

    return __a, __b, __c


temp_m, temp_q = find_linear_equation(1.6, 1.024, 2.5, 2)

co2_m, co2_q = find_linear_equation(outdoor_co2, max_reward_co2, acceptable_co2, min_reward_co2)
co2_a, co2_b, co2_c = find_parabola_with_discriminant_zero(acceptable_co2, min_reward_co2, ideal_max_co2)
voc_m, voc_q = find_linear_equation(outdoor_voc, max_reward_voc, acceptable_voc, min_reward_voc)
voc_a, voc_b, voc_c = find_parabola_with_discriminant_zero(acceptable_voc, min_reward_voc, ideal_max_voc)

# Reward factors
air_quality_factor = 1
comfort_factor = 1
energy_factor = 0.1
# ---

# Later imports for initializations
from state_data import StateData
from action_data import ActionData
from environment import Environment

# These variables are initialized automatically, don't change them ---
last_hour: int
last_minute: int
line_count: int  # Number of steps of mcts simulation (i.e., lines in reservation profile string)

env: Environment


stats_path: str
stats_filename: str
timestamp: str

step_stats_folder = 'step_stats/'

reservations: defaultdict
referenced_reservations: defaultdict
# ---


def set_step_stats(value) -> None:
    """
    Replaces the value of `utils.step_stats` with the one provided.

    Args:
        value: new value.
    """
    global step_stats
    step_stats = value



def set_time_delta(value) -> None:
    """
    Replaces the value of `utils.time_delta` with the one provided.

    Args:
        value: new value.
    """
    global time_delta
    time_delta = value


def set_initial_temp_in(new_value: float) -> None:
    """
    Replaces the value of `utils.initial_temp_in` with the one provided.

    Args:
        new_value: new value.
    """
    global initial_temp_in
    initial_temp_in = new_value


def initialize_all(environment: Environment) -> None:
    """
    Initializes the framework. In particular, sets up the `utils.model_param` and `utils.env` variables so that are
    available to access outside the `utils.py` file.

    Furthermore, if you set `utils.step_stats` to True, it enables the statistics recording system.

    Finally, checks if there are duplicate actions in `action_data.py`.

    Args:
        model_parameters: the MCTS parameters.
        environment: a general environment that follows the rules of `Environment` class.
    """
    global env

    env = environment

    if step_stats:
        initialize_stats()

    # Checks ---
    # Check action codes
    codes = []
    for action in ActionData:
        if action.code not in codes:
            codes.append(action.code)
        else:
            error('(ActionData) Error! Code number of `%s` action is already assigned.' % action.name)
    # ---


def initialize_stats() -> None:
    """
    Function exclusively used by `utils.initialize_all()`. It enables the statistics recording system.
    """
    global stats_path
    global stats_filename
    global timestamp

    stats_path, stats_filename, timestamp = stats.create_stats_path_and_filename()
    if verbose:
        print('Stats file location: ' + stats_path + stats_filename)


class Reservations:
    """
    Class exclusively used by `utils.get_reservation_dict()` to store reservations files data.
    """
    def __init__(self, people, temp_out) -> None:
        """
        Method called after the creation of the object. It just saves the data.

        Args:
            people: number of people inside the room at a given time.
            temp_out: forecasted outdoor temperature at a given time.
        """
        if people > max_people:
            error('(get_reservations_profile) csv file exceeded `max_people` (%d) threshold.' % max_people)
        self.__people = people
        self.__temp_out = temp_out

    @property
    def people(self) -> int:
        """
        Property method of `Reservations` class.

        Returns: the number of people inside the room at a given time.
        """
        return self.__people

    @property
    def temp_out(self) -> float:
        """
        Property method of `Reservations` class.

        Returns: forecasted outdoor temperature at a given time.
        """
        return self.__temp_out


def get_reservation_dict(csv_reader) -> defaultdict[dict]:
    """
    Exclusively used by `utils.get_reservations_profile()`. Returns the room reservations as a dict given a csv reader
    created by `utils.get_reservations_profile()`.

    Args:
        csv_reader: csv reader created by `utils.get_reservations_profile()`.

    Returns: the room reservations dict.
    """
    global line_count

    line_count = 0
    reservations = defaultdict(dict)
    for row in csv_reader:
        date = row['time']
        people = int(row['#people'])
        temp_out = float(row['temp_out'])

        date_obj = datetime.strptime(date, '%H:%M')
        hour = date_obj.hour
        minute = date_obj.minute

        if hour not in reservations:
            new_dict = defaultdict(dict)
            reservations[hour] = new_dict

        reservations[hour][minute] = Reservations(people, temp_out)
        line_count += 1

    if line_count == 0:
        error('Reservation data is empty.')

    return reservations


def get_reservations_profile(reservations_filepath: str = None, string: str = None) -> defaultdict[dict]:
    """
    Calls `utils.get_reservation_dict()` providing the right parameter (i.e., `reservations_filepath` or `string`) as
    its input.

    Args:
        reservations_filepath: location of the reservations file in the file system. Must be a csv file. It cannot be
            ignored if `string` parameter already is.
        string: string possibly generated by `utils.generate_reservations_file()`. It cannot be ignored if
            `reservations_filepath` parameter already is.

    Returns: the dict created by `utils.get_reservation_dict()`.
    """
    if string is not None:
        csv_lines = string.splitlines()
        csv_reader = csv.DictReader(csv_lines)

        return get_reservation_dict(csv_reader)

    try:
        with open(reservations_filepath) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            if verbose:
                print('Initialization of the room\'s reservations from file.')
                print('File: ' + reservations_filepath)
                print('Column names are ' + str(csv_reader.fieldnames) + '.')

            reservations = get_reservation_dict(csv_reader)

    except FileNotFoundError:
        error('Reservations file does not exist!')

    return reservations


def generate_index_key_references(dictionary) -> defaultdict[dict]:
    """
    Creates a referenced dict for keys and indexes of the given dict.

    Args:
        dictionary: the dict provided in input.

    Returns: the referenced dict.
    """
    referenced_dict = defaultdict(dict)
    ki = defaultdict(int)
    ik = defaultdict(int)
    for i, k in enumerate(dictionary):
        ki[k] = i  # dictionary index_of_key
        ik[i] = k  # dictionary key_of_index

    referenced_dict['ki'] = ki
    referenced_dict['ik'] = ik
    return referenced_dict


def get_referenced_reservations() -> defaultdict[dict]:
    """
    Calls `utils.generate_index_key_references()` two times: the first for the hours and the second for the minutes of
    the room reservations dict. Creating this additional dict is useful when we want to quickly get the next hour and
    next minute given the current time. It is only used in `SafePlaceEnv.next_time()`.

    Returns: the referenced dicts.
    """
    referenced_reservations = defaultdict(dict)

    referenced_reservations['hours'] = generate_index_key_references(reservations)

    referenced_reservations['minutes'] = defaultdict(dict)
    for hour in reservations.keys():
        referenced_dict = generate_index_key_references(reservations[hour])
        referenced_reservations['minutes'][hour] = referenced_dict

    return referenced_reservations


def update_reservations(reservations_filepath: str = None, string: str = None, random_initial_temp_in: bool = True) \
        -> None:
    """
    Only this function gets called outside `utils` module to obtain and save the room reservations dict on a given day
    and its related referenced dicts. These dicts will be used by MCTS throughout the simulation (look at
    `safeplace_env.py`).

    Furthermore, this function randomly chooses an initial indoor temperature given the initial outdoor temperature.

    Args:
        reservations_filepath: location of the reservations file in the file system. Must be a csv file. It cannot be
            ignored if `string` parameter already is.
        string: string possibly generated by `utils.generate_reservations_file()`. It cannot be ignored if
            `reservations_filepath` parameter already is.
        random_initial_temp_in: set to True if you want a randomized `utils.initial_temp_in`; otherwise, don't change
            the value already given to `utils.initial_temp_in`. Default to True.
    """
    global reservations
    global referenced_reservations
    global initial_temp_in

    if reservations_filepath is None and string is None:
        error('Both `reservations_filepath` and `string` are `None`.')
    if reservations_filepath is not None and string is not None:
        error('Both `reservations_filepath` and `string` are not `None`.')

    reservations = get_reservations_profile(reservations_filepath=reservations_filepath, string=string)
    referenced_reservations = get_referenced_reservations()

    if random_initial_temp_in:
        init_min = random_initial_temp_in_min
        init_max = random_initial_temp_in_max
        out_min = reservations_generation_min_temp
        out_max = reservations_generation_max_temp
        threshold = random_initial_temp_in_threshold

        first_hour = min(reservations.keys())
        first_minute = min(reservations[first_hour].keys())

        initial_temp_in = random.uniform(init_min, init_max)

        if out_min <= reservations[first_hour][first_minute].temp_out < init_min \
                and (init_max - threshold) < initial_temp_in <= init_max:

            initial_temp_in = random.uniform(init_min, init_max - threshold)

        elif init_max < reservations[first_hour][first_minute].temp_out <= out_max \
                and init_min <= initial_temp_in < (init_min + threshold):

            initial_temp_in = random.uniform(init_min + threshold, init_max)


def initial_state() -> StateData:
    """
    Creates the initial `StateData` object that represents the initial state of the MDP given the room reservations and
    other static parameters such as `utils.initial_co2` and `utils.initial_voc`.

    Returns: the initial `StateData` object.
    """
    global last_hour
    global last_minute

    first_hour = min(reservations.keys())
    first_minute = min(reservations[first_hour].keys())
    people = reservations[first_hour][first_minute].people
    co2 = initial_co2
    voc = initial_voc
    # First observations
    temp_in = initial_temp_in
    temp_out = reservations[first_hour][first_minute].temp_out

    last_hour = max(reservations.keys())
    last_minute = max(reservations[last_hour].keys())

    return StateData(first_hour, first_minute, people, co2, voc, temp_in, temp_out)


def set_verbose(new_value: bool) -> None:
    """
    Replaces the value of `utils.verbose` with the one provided.

    Args:
        new_value: new value.
    """
    global verbose
    verbose = new_value


def error(text: str) -> None:
    """
    Prints a red-colored error message and exits the program.

    Args:
        text: string to print to the terminal.
    """
    print('\033[91m' + text + '\033[0m')
    sys.exit(1)


def warning(text: str) -> None:
    """
    Prints a yellow-colored warning message.

    Args:
        text: string to print to the terminal.
    """
    print('\033[93m' + text + '\033[0m')


# Last settings and print info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
