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
import copy

import numpy as np
import tensorflow as tf
import execnet

import utils
from neural_network import NN, reset_weights, pypy_pickle, polyak_average
import execnet_pypy


def load_external_nn_data():
    # `model_path` and `dataset_path` are the only allowed variables to be assigned in this function
    if utils.external_model_path is None:
        utils.error('(nn) Assign a valid `external_model_path`.')
    nn = NN(model_path=utils.external_model_path, dataset_path=utils.external_dataset_path, shuffle=True)

    model_name = 'safeplace_simulated_nn_%s_%d_%s' % (str(nn.layers), utils.already_completed_epochs, nn.timestamp)
    initial_pickle_path = pypy_pickle(nn, model_name)

    return nn, initial_pickle_path, utils.already_completed_epochs


def train():
    if isinstance(utils.tf_seed, int):
        tf.random.set_seed(utils.tf_seed)
        np.random.seed(utils.tf_seed)
        utils.warning('(nn) Tensorflow seed: %d' % utils.tf_seed)
    else:
        utils.warning('(nn) `tf_seed` is not an int value. The experiment will not be reproducible.')

    # TODO: ask this to the user
    pypy3_path = utils.execnet_interpreter_path
    config = 'popen//python=' + pypy3_path
    dataset_path = 'datasets/expert_dataset/expert_dataset.csv'
    layers = (15, 30, 40)

    already_completed_epochs = 0
    initial_epochs = 1000
    initial_batch_size = 64
    epochs = 500
    batch_size = 8
    simulations = 1
    batches = 100

    if not utils.skip_initial_training:
        nn = NN(dataset_path=dataset_path,
                layers=layers,
                shuffle=True)
        timestamp = nn.timestamp

        model_name = 'safeplace_simulated_nn_%s_%d_%s' % (str(layers), initial_epochs, timestamp)
        model_path = utils.nn_folder + model_name + '.h5'

        # TODO: change to fit_v2 and make a custom model
        # https://www.tensorflow.org/guide/keras/custom_layers_and_models#the_model_class
        # https://www.tensorflow.org/guide/keras/save_and_serialize
        nn.fit(epochs=initial_epochs, batch_size=initial_batch_size, save_best=True, filepath=model_path)

        # Load best model
        nn.model = tf.keras.models.load_model(model_path)
        initial_nn_pickle_path = pypy_pickle(nn, model_name)

        if utils.initial_training_only:
            if utils.verbose:
                print('(nn) Finished initial training. Exiting...')
            quit()
    else:
        initial_epochs = 0
        nn, initial_nn_pickle_path, already_completed_epochs = load_external_nn_data()
        timestamp = nn.timestamp
        layers = nn.layers

    datasets = (utils.deploy_folder + 'fd_%s.csv' % timestamp,
                utils.deploy_folder + 'vd_%s.csv' % timestamp)

    if utils.polyak:
        previous_nn = NN(model=copy.deepcopy(nn.model))
    elif utils.use_two_neural_networks:
        reset_weights(nn.model)

    gw = execnet.makegateway(config)
    channel = gw.remote_exec(source=execnet_pypy)
    # Sending `execnet_pypy.mcts` function args
    channel.send((batches, simulations, datasets, timestamp))

    channel.send(initial_nn_pickle_path)

    log_folder = None
    writer = None

    if utils.verbose:
        print('(nn) Waiting for first data batch...', flush=True)
    for batch in range(batches):
        transition_data: list = channel.receive()
        if utils.verbose:
            print('(nn) Data received. Starting training n.%d...' % (batch + 1), flush=True)

        # Arranging data already scrambled
        fd_list: list = transition_data[0]
        vd_list: list = transition_data[1]

        def from_list_to_np(d_list: list):
            string = ''.join(d_list)
            string = string[:-1]
            string = string.replace('\n', ';')
            d = np.matrix(string)
            x = d[:, :-3]
            y = d[:, -3:]
            x = np.array(x)
            y = np.array(y)

            return x, y

        fd_x, fd_y = from_list_to_np(fd_list)
        vd_x, vd_y = from_list_to_np(vd_list)

        if utils.tf_dataset:
            if not utils.optimize_dataset_between_batches:
                fd = tf.data.Dataset.from_tensor_slices((fd_x.tolist(), fd_y.tolist()))
                vd = tf.data.Dataset.from_tensor_slices((vd_x.tolist(), vd_y.tolist()))

                if batch == 0 and utils.discard_initial_dataset:
                    fd = nn.load_single_dataset(datasets[0])
                    vd = nn.load_single_dataset(datasets[1])

                    nn.set_fit_data(fd)
                    nn.set_validation_data(vd)
                elif utils.polyak:
                    nn.set_fit_data(fd)
                    nn.set_validation_data(vd)
                else:
                    nn.set_fit_data(nn.fit_data.concatenate(fd))
                    nn.set_validation_data(nn.validation_data.concatenate(vd))
            else:
                nn.update_from_support_datasets(fd_x, fd_y, vd_x, vd_y)
        else:
            if utils.optimize_dataset_between_batches:
                utils.error('(nn) Set `tf_dataset` to True. Legacy mode not supported.')

            if batch == 0 and utils.discard_initial_dataset:
                fd = nn.load_single_dataset(datasets[0])
                vd = nn.load_single_dataset(datasets[1])

                nn.set_fit_data(fd)
                nn.set_validation_data(vd)
            elif utils.polyak:
                nn.set_fit_data((fd_x, fd_y))
                nn.set_validation_data((vd_x, vd_y))
            else:
                fd = (np.concatenate((nn.fit_data[0], fd_x)), np.concatenate((nn.fit_data[1], fd_y)))
                nn.set_fit_data(fd)

                vd = (np.concatenate((nn.validation_data[0], vd_x)), np.concatenate((nn.validation_data[1], vd_y)))
                nn.set_validation_data(vd)

        if utils.verbose:
            print('(nn) Fit dataset length: %d' % len(nn.fit_data))
            print('(nn) Validation dataset length: %d' % len(nn.validation_data))

        current_total_epochs = already_completed_epochs + initial_epochs + (batch + 1) * epochs
        model_name = 'safeplace_nn_%s_%d_%s' % (str(layers), current_total_epochs, timestamp)
        model_path = utils.nn_folder + model_name + '.h5'

        # nn.fit(epochs=epochs, batch_size=batch_size, save_best=True, filepath=model_path)

        # this version does not save the model
        """if batch == 1:
            log_folder = 'tf_logs/' + timestamp + '/'
            # tf.profiler.experimental.start(log_folder)
            writer = tf.summary.create_file_writer(log_folder)"""
        nn.fit_v2_tf(epochs=epochs, batch_size=batch_size, save_best=True, filepath=model_path,
                     writer=writer, log_folder=log_folder)
        """if batch == 1:
            tf.profiler.experimental.stop()
            quit()"""

        # Load best model of the last number of `epochs`
        # nn.model = tf.keras.models.load_model(model_path)

        utils.warning(str(get_current_memory_usage()))
        print(end='', flush=True)

        if utils.polyak:
            polyak_average(nn, previous_nn)

        pickle_path = pypy_pickle(nn, model_name)

        if batch != batches - 1:
            channel.send(pickle_path)
            if utils.verbose:
                print('(nn) New weights location sent.', flush=True)

    if utils.verbose:
        print('(nn) All processes have finished their tasks! Exiting...', flush=True)


if __name__ == '__main__':
    def get_current_memory_usage():
        with open('/proc/self/status') as f:
            memusage = f.read().split('VmRSS:')[1].split('\n')[0][:-3]

        return int(memusage.strip()) / 1000

    if not utils.execnet:
        utils.error('(execnet) Set `adaptability_execnet` to True.')

    if utils.polyak and not utils.discard_initial_dataset:
        utils.error('(execnet) If `polyak` is set to True, also `discard_initial_dataset` must be set to True.')

    if utils.compare_with_oracle and not utils.batch_stats:
        utils.error('(execnet) If `compare_with_oracle` is set to True, also `batch_stats` must be set to True.')

    if not utils.skip_initial_training and utils.initial_epochs <= 0:
        utils.error('(execnet) If `skip_initial_training` is set to False, `initial_epochs` must be greater than 0.')

    if utils.optimize_dataset_between_batches and utils.external_dataset_path is None:
        utils.error('(execnet) If `optimize_dataset_between_batches` is set to True, provide `external_dataset_path`.')

    start = time()
    train()
    end = time()
    print('TOTAL time elapsed: ' + str(end - start))
