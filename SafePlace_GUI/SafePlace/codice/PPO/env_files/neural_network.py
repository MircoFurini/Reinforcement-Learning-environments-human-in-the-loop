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
import gc
import pickle
from datetime import datetime
import time
from math import ceil

import numpy as np
import pandas as pd
import tensorflow as tf

import utils
from matrix_pypy import NNPyPy


class NN:
    def __init__(self, model: tf.keras.Model = None, model_path: str = None,
                 datasets: tuple = None, dataset_path: str = None,
                 layers: tuple = None, shuffle: bool = True):
        if utils.verbose:
            print('(NN) `shuffle` is set to %s.' % str(shuffle))

        self.timestamp = datetime.utcnow().strftime('%y%m%d_%H%M%S')

        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.fit_metric = tf.keras.metrics.MeanSquaredError()
        self.val_metric = tf.keras.metrics.MeanSquaredError()

        self.fd_xs = None
        self.fd_ys = None
        self.vd_xs = None
        self.vd_ys = None

        if model is not None and model_path is not None:
            utils.error('(NN) `model` and `model_path` can\'t be both assigned.')
        if datasets is not None and dataset_path is not None:
            utils.error('(NN) `datasets` and `dataset_path` can\'t be both assigned.')
        if model is None and model_path is None:
            if datasets is None and dataset_path is None:
                utils.error('(NN) when the model is not retrievable, `datasets` or `dataset_path` (exclusive or) is '
                            'needed.')

            if layers is None:
                self.layers = (15, 30, 40)
                if utils.verbose:
                    print('(NN) Automatically set neural network layers to %s.' % str(self.layers))
            else:
                if not len(layers) > 0:
                    utils.error('(NN) Don\'t provide an empty `layer` tuple.')
                self.layers = layers

            self.is_normalized = True
            self.model = self.create_model()

            # Legacy
            if not utils.tf_dataset:
                self.model.compile(loss='mean_squared_error', optimizer='adam')

            if datasets is None:
                self.fit_data, self.validation_data, self.test_data = \
                    self.load_data_and_normalize(dataset_path=dataset_path, shuffle=shuffle, timestamp=self.timestamp)
            else:
                self.fit_data, self.validation_data, self.test_data = datasets

            # Legacy
            if not utils.tf_dataset:
                norm = self.model.get_layer('normalization')
                norm.adapt(self.fit_data[0])
        else:
            if model is not None:
                if not isinstance(model, tf.keras.Model):
                    utils.error('(NN) `model` is not a keras model.')
                self.model = model

            if model_path is not None:
                self.model = tf.keras.models.load_model(model_path)

            if layers is not None:
                utils.warning('(NN) Why is `layers` assigned? The program obtains the layers information by itself.')

            self.is_normalized = False
            layers = []
            for idx, l in enumerate(self.model.layers):
                if idx == 0:
                    continue
                elif idx == 1 and l.name == 'normalization':
                    self.is_normalized = True
                    continue
                elif idx == len(self.model.layers) - 1:
                    continue
                else:
                    layers.append(l.output_shape[1])
            self.layers = tuple(layers)

            # Load a possible dataset
            if dataset_path is not None:
                self.fit_data, self.validation_data, self.test_data = \
                    self.load_data_and_normalize(dataset_path=dataset_path, shuffle=shuffle, timestamp=self.timestamp)
            elif datasets is not None:
                self.fit_data, self.validation_data, self.test_data = datasets
            else:
                if not (utils.execnet and utils.polyak):
                    NN.initialize_csv_datasets(self.timestamp)
                self.fit_data, self.validation_data, self.test_data = None, None, None

    @tf.function
    def train_step(self, x, y):
        print('(train_step) Retracing...', flush=True)
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss_value = self.loss_fn(y, logits)

        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.fit_metric.update_state(y, logits)
        # tf.print('model variables: ' + str(sys.getsizeof(self.model.variables)))
        # tf.print('model weights: ' + str(sys.getsizeof(self.model.weights)))
        # tf.print('model history: ' + str(sys.getsizeof(self.model.history)))

        # tf.print('tape: ' + str(sys.getsizeof(tape._tape)))
        # tf.print('tape watched variables: ' + str(sys.getsizeof(tape._watched_variables)))
        return loss_value

    @tf.function
    def test_step(self, x, y):
        print('(test_step) Retracing...', flush=True)
        val_logits = self.model(x, training=False)
        self.val_metric.update_state(y, val_logits)

    def fit_v2_tf(self, epochs: int, batch_size: int, fit_verbose: int = 2, save_best: bool = True,
                  filepath: str = None, writer=None, log_folder=None):
        min_val_loss = float('+inf')
        best_weights = None

        fit_dataset = self.fit_data.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE,
                                          deterministic=False).prefetch(20)

        validation_dataset = self.validation_data.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE,
                                                        deterministic=False).prefetch(20)
        step_num = 0

        for epoch in range(epochs):
            print('Epoch %d/%d - Batches: %d' % (epoch + 1, epochs, len(fit_dataset)))
            start = time.time()

            if fit_verbose == 1:
                utils.warning('before training steps:')
                utils.warning(str(get_current_memory_usage()))

            if log_folder is not None:
                _fit_dataset = iter(fit_dataset)
                for step in range(len(fit_dataset)):
                    step_num += 1
                    with tf.profiler.experimental.Trace('train', step_num=step_num, _r=1):
                        x_fit_batch, y_fit_batch = next(_fit_dataset)
                        loss_value = self.train_step(x_fit_batch, y_fit_batch)
            else:
                for step, (x_fit_batch, y_fit_batch) in enumerate(fit_dataset):
                    loss_value = self.train_step(x_fit_batch, y_fit_batch)

            if fit_verbose == 1:
                utils.warning('after training steps:')
                utils.warning(str(get_current_memory_usage()))

            loss = self.fit_metric.result()
            loss = float(loss)
            self.fit_metric.reset_states()

            for x_val_batch, y_val_batch in validation_dataset:
                self.test_step(x_val_batch, y_val_batch)

            if fit_verbose == 1:
                utils.warning('after validation steps:')
                utils.warning(str(get_current_memory_usage()))

            val_loss = self.val_metric.result()
            val_loss = float(val_loss)
            self.val_metric.reset_states()
            if fit_verbose <= 2:
                print('Loss: %.4f - Validation loss: %.4f - Time elapsed: %.2fs' % (loss, val_loss,
                                                                                    time.time() - start))
            if save_best and val_loss < min_val_loss:
                if fit_verbose <= 2:
                    print('Current validation loss is lower than %.4f. Saving and overwriting best model weights.' %
                          min_val_loss)
                min_val_loss = val_loss
                best_weights = self.model.get_weights()

        if save_best:
            self.model.set_weights(best_weights)

    def fit_v2_numpy(self, epochs: int, batch_size: int, fit_verbose: int = 2, save_best: bool = True,
                     filepath: str = None, writer=None, log_folder=None):
        min_val_loss = float('+inf')
        best_weights = None

        fit_batches = ceil(self.fit_data[0].shape[0] / batch_size)
        validation_batches = ceil(self.validation_data[0].shape[0] / batch_size)

        x_fit_batches = np.array_split(self.fit_data[0], fit_batches)
        y_fit_batches = np.array_split(self.fit_data[1], fit_batches)
        x_validation_batches = np.array_split(self.validation_data[0], validation_batches)
        y_validation_batches = np.array_split(self.validation_data[1], validation_batches)

        for epoch in range(epochs):
            print("Epoch %d/%d - Batches: %d" % (epoch + 1, epochs, fit_batches))
            start = time.time()

            if fit_verbose == 1:
                utils.warning('before training steps:')
                utils.warning(str(get_current_memory_usage()))

            # Profiler experimental
            if log_folder is not None:
                _x_fit_batches = iter(x_fit_batches)
                _y_fit_batches = iter(y_fit_batches)

                for step in range(fit_batches):
                    with tf.profiler.experimental.Trace('train', step_num=(step + 1) * (epoch + 1), _r=1):
                        x_fit_batch = next(_x_fit_batches)
                        y_fit_batch = next(_y_fit_batches)
                        x_fit_batch = tf.convert_to_tensor(x_fit_batch, dtype=tf.float32)
                        y_fit_batch = tf.convert_to_tensor(y_fit_batch, dtype=tf.float32)
                        loss_value = self.train_step(x_fit_batch, y_fit_batch)
            else:
                for step, (x_fit_batch, y_fit_batch) in enumerate(zip(x_fit_batches, y_fit_batches)):
                    x_fit_batch = tf.convert_to_tensor(x_fit_batch, dtype=tf.float32)
                    y_fit_batch = tf.convert_to_tensor(y_fit_batch, dtype=tf.float32)
                    loss_value = self.train_step(x_fit_batch, y_fit_batch)

            if fit_verbose == 1:
                utils.warning('after training steps:')
                utils.warning(str(get_current_memory_usage()))

            loss = self.fit_metric.result()
            loss = float(loss)
            self.fit_metric.reset_states()

            for x_val_batch, y_val_batch in zip(x_validation_batches, y_validation_batches):
                x_val_batch = tf.convert_to_tensor(x_val_batch, dtype=tf.float32)
                y_val_batch = tf.convert_to_tensor(y_val_batch, dtype=tf.float32)
                self.test_step(x_val_batch, y_val_batch)

            if fit_verbose == 1:
                utils.warning('after validation steps:')
                utils.warning(str(get_current_memory_usage()))

            val_loss = self.val_metric.result()
            val_loss = float(val_loss)
            self.val_metric.reset_states()
            if fit_verbose <= 2:
                print('Loss: %.4f - Validation loss: %.4f - Time elapsed: %.2fs' % (loss, val_loss,
                                                                                    time.time() - start))
            if save_best and val_loss < min_val_loss:
                if fit_verbose <= 2:
                    print('Current validation loss is lower than %.4f. Saving and overwriting best model weights.' %
                          min_val_loss)
                min_val_loss = val_loss
                best_weights = self.model.get_weights()

        if save_best:
            self.model.set_weights(best_weights)

    def create_model(self) -> tf.keras.Model:
        input = tf.keras.Input(shape=8, name='input')
        h = input
        h = tf.keras.layers.Normalization()(h)

        for i in range(len(self.layers)):
            h = tf.keras.layers.Dense(self.layers[i], activation='relu', name='hidden_' + str(i))(h)

        y = tf.keras.layers.Dense(3, activation='linear', name='output')(h)
        model = tf.keras.Model(inputs=input, outputs=y)

        return model

    def load_data_and_normalize(self, dataset_path: str, shuffle: bool, timestamp: str) -> \
            tuple[tf.data.Dataset, tf.data.Dataset, None]:

        fd = pd.read_csv(dataset_path)

        if shuffle:
            fd = fd.sample(frac=1)

        rows_to_remove = int(fd.shape[0] * utils.dataset_factor)

        fd, vd = fd.drop(fd.head(rows_to_remove).index), fd.head(rows_to_remove)
        fd, vd = fd.reset_index(drop=True), vd.reset_index(drop=True)

        if utils.execnet:
            if not os.path.exists(utils.deploy_folder):
                os.makedirs(utils.deploy_folder)

            if utils.discard_initial_dataset:
                header = fd.drop(fd.index.to_list()[:])
                header.to_csv(utils.deploy_folder + 'fd_%s.csv' % timestamp, index=False)
                header.to_csv(utils.deploy_folder + 'vd_%s.csv' % timestamp, index=False)
            else:
                fd.to_csv(utils.deploy_folder + 'fd_%s.csv' % timestamp, index=False)
                vd.to_csv(utils.deploy_folder + 'vd_%s.csv' % timestamp, index=False)

        fd_x = fd.copy()
        fd_y = fd_x[['next_co2', 'next_voc', 'next_temp_in']].copy()
        fd_x = fd_x.drop(['next_co2', 'next_voc', 'next_temp_in'], axis=1)

        vd_x = vd.copy()
        vd_y = vd_x[['next_co2', 'next_voc', 'next_temp_in']].copy()
        vd_x = vd_x.drop(['next_co2', 'next_voc', 'next_temp_in'], axis=1)

        if utils.optimize_dataset_between_batches:
            if not utils.tf_dataset:
                utils.error('(NN) Set `tf_dataset` to True. Legacy mode not supported.')

            self.fd_xs, self.fd_ys, self.vd_xs, self.vd_ys = NN.add_expert_column(fd_x, fd_y, vd_x, vd_y, value=1)

            if utils.execnet:
                return None, None, None

        if utils.tf_dataset:
            fit_data_x = tf.data.Dataset.from_tensor_slices(fd_x.values.tolist()).batch(50000)
            norm = self.model.get_layer('normalization')
            norm.adapt(fit_data_x)

            if not utils.execnet and utils.optimize_dataset_between_batches:
                return None, None, None

            fit_data = tf.data.Dataset.from_tensor_slices((fd_x.values.tolist(), fd_y.values.tolist()))
            validation_data = tf.data.Dataset.from_tensor_slices((vd_x.values.tolist(), vd_y.values.tolist()))

            fit_data = fit_data.shuffle(buffer_size=10000, seed=utils.tf_seed, reshuffle_each_iteration=True)
            validation_data = validation_data.shuffle(buffer_size=10000, seed=utils.tf_seed, reshuffle_each_iteration=True)
        else:
            # Legacy
            fit_data = (fd_x.to_numpy(), fd_y.to_numpy())
            validation_data = (vd_x.to_numpy(), vd_y.to_numpy())

        return fit_data, validation_data, None

    @staticmethod
    def add_expert_column(fd_x, fd_y, vd_x, vd_y, value):
        """
        Only `fd_x` and `vd_x` get edited.
        """

        fd_expert = [value for _ in range(len(fd_x))]
        fd_x['expert'] = fd_expert

        if value == 1:
            vd_expert = fd_expert[:len(vd_x)]
            vd_x['expert'] = vd_expert

        return fd_x, fd_y, vd_x, vd_y

    def update_from_support_datasets(self, new_fd_x, new_fd_y, new_vd_x, new_vd_y):
        # step 1: delete old data inside support datasets following a given criterion
        fd_xs_people = self.fd_xs['people']
        fd_xs_co2 = self.fd_xs['co2']
        fd_xs_voc = self.fd_xs['voc']
        fd_xs_temp_in = self.fd_xs['temp_in']
        fd_xs_temp_out = self.fd_xs['temp_out']
        fd_xs_window_open = self.fd_xs['window_open']
        fd_xs_ach = self.fd_xs['ach']
        fd_xs_sanitizer_active = self.fd_xs['sanitizer_active']
        fd_xs_expert = self.fd_xs['expert']

        vd_xs_people = self.vd_xs['people']
        vd_xs_co2 = self.vd_xs['co2']
        vd_xs_voc = self.vd_xs['voc']
        vd_xs_temp_in = self.vd_xs['temp_in']
        vd_xs_temp_out = self.vd_xs['temp_out']
        vd_xs_window_open = self.vd_xs['window_open']
        vd_xs_ach = self.vd_xs['ach']
        vd_xs_sanitizer_active = self.vd_xs['sanitizer_active']
        vd_xs_expert = self.vd_xs['expert']

        """for np_idx, row in enumerate(new_fd_x):
            fd_action_zip = zip(fd_xs_window_open, fd_xs_ach, fd_xs_sanitizer_active)
            fd_state_zip = zip(fd_xs_people, fd_xs_co2, fd_xs_voc, fd_xs_temp_in, fd_xs_temp_out)

            new_state = (row[0], row[1], row[2], row[3], row[4])
            new_action = (row[5], row[6], row[7])

            found_one_entry = False

            for df_idx, action, state, expert in zip(self.fd_xs.index, fd_action_zip, fd_state_zip, fd_xs_expert):
                if expert == 1 and action == new_action:
                    if utils.states_are_close(new_state, state):
                        if not found_one_entry:
                            self.fd_xs.loc[df_idx, :] = new_state + new_action + (0,)
                            self.fd_ys.loc[df_idx, :] = new_fd_y[np_idx]
                            found_one_entry = True
                        else:
                            self.fd_xs = self.fd_xs.drop(df_idx)
                            self.fd_ys = self.fd_ys.drop(df_idx)"""

        def update_dataframe(d_xs: pd.DataFrame, d_ys: pd.DataFrame, new_d_x: np.ndarray, new_d_y: np.ndarray,
                             new_other_d_x: np.ndarray):
            d_xs_people = d_xs['people']
            d_xs_co2 = d_xs['co2']
            d_xs_voc = d_xs['voc']
            d_xs_temp_in = d_xs['temp_in']
            d_xs_temp_out = d_xs['temp_out']
            d_xs_window_open = d_xs['window_open']
            d_xs_ach = d_xs['ach']
            d_xs_sanitizer_active = d_xs['sanitizer_active']
            d_xs_expert = d_xs['expert']

            d_action_zip = zip(d_xs_window_open, d_xs_ach, d_xs_sanitizer_active)
            d_state_zip = zip(d_xs_people, d_xs_co2, d_xs_voc, d_xs_temp_in, d_xs_temp_out)

            def delete_dataframe_rows(d_xs, d_ys, array):
                pass

            for df_idx, action, state, expert in zip(d_xs.index, d_action_zip, d_state_zip, d_xs_expert):
                for np_idx, row in enumerate(new_d_x):
                    new_state = (row[0], row[1], row[2], row[3], row[4])
                    new_action = (row[5], row[6], row[7])

                    if expert == 1 and action == new_action:
                        if utils.states_are_close(new_state, state):
                            try:
                                d_xs = d_xs.drop(df_idx)
                                d_ys = d_ys.drop(df_idx)
                            except KeyError:
                                pass
                            break

                for np_idx, row in enumerate(new_other_d_x):
                    new_state = (row[0], row[1], row[2], row[3], row[4])
                    new_action = (row[5], row[6], row[7])

                    if expert == 1 and action == new_action:
                        if utils.states_are_close(new_state, state):
                            try:
                                d_xs = d_xs.drop(df_idx)
                                d_ys = d_ys.drop(df_idx)
                            except KeyError:
                                pass
                            break

            # insert new data
            new_d_x = pd.DataFrame(new_d_x, columns=d_xs.columns[:-1])
            new_d_y = pd.DataFrame(new_d_y, columns=d_ys.columns)
            new_d_x, _, _, _ = NN.add_expert_column(new_d_x, None, None, None, value=0)

            d_xs = d_xs.append(new_d_x)
            d_ys = d_ys.append(new_d_y)
            d_xs = d_xs.reset_index(drop=True)
            d_ys = d_ys.reset_index(drop=True)

            return d_xs, d_ys

        self.fd_xs, self.fd_ys = update_dataframe(self.fd_xs, self.fd_ys, new_fd_x, new_fd_y, new_vd_x)

        """fd_action_zip = zip(fd_xs_window_open, fd_xs_ach, fd_xs_sanitizer_active)
        fd_state_zip = zip(fd_xs_people, fd_xs_co2, fd_xs_voc, fd_xs_temp_in, fd_xs_temp_out)

        row_inserted = [False for _ in range(len(new_fd_x))]

        for df_idx, action, state, expert in zip(self.fd_xs.index, fd_action_zip, fd_state_zip, fd_xs_expert):
            for np_idx, row in enumerate(new_fd_x):
                new_state = (row[0], row[1], row[2], row[3], row[4])
                new_action = (row[5], row[6], row[7])

                if expert == 1 and action == new_action:
                    if utils.states_are_close(new_state, state):
                        if not row_inserted[np_idx]:
                            self.fd_xs.loc[len(self.fd_xs)] = new_state + new_action + (0,)
                            self.fd_ys.loc[len(self.fd_ys)] = new_fd_y[np_idx]
                            row_inserted[np_idx] = True

                        try:
                            self.fd_xs = self.fd_xs.drop(df_idx)
                            self.fd_ys = self.fd_ys.drop(df_idx)
                        except KeyError:
                            pass

            for np_idx, row in enumerate(new_vd_x):
                new_state = (row[0], row[1], row[2], row[3], row[4])
                new_action = (row[5], row[6], row[7])

                if expert == 1 and action == new_action:
                    if utils.states_are_close(new_state, state):
                        try:
                            self.fd_xs = self.fd_xs.drop(df_idx)
                            self.fd_ys = self.fd_ys.drop(df_idx)
                        except KeyError:
                            pass

        # vd
        vd_action_zip = zip(vd_xs_window_open, vd_xs_ach, vd_xs_sanitizer_active)
        vd_state_zip = zip(vd_xs_people, vd_xs_co2, vd_xs_voc, vd_xs_temp_in, vd_xs_temp_out)

        row_inserted = [False for _ in range(len(new_vd_x))]

        for df_idx, action, state, expert in zip(self.vd_xs.index, vd_action_zip, vd_state_zip, vd_xs_expert):
            for np_idx, row in enumerate(new_vd_x):
                new_state = (row[0], row[1], row[2], row[3], row[4])
                new_action = (row[5], row[6], row[7])

                if expert == 1 and action == new_action:
                    if utils.states_are_close(new_state, state):
                        if not row_inserted[np_idx]:
                            self.vd_xs.loc[len(self.vd_xs)] = new_state + new_action + (0,)
                            self.vd_ys.loc[len(self.vd_ys)] = new_vd_y[np_idx]
                            row_inserted[np_idx] = True

                        try:
                            self.vd_xs = self.vd_xs.drop(df_idx)
                            self.vd_ys = self.vd_ys.drop(df_idx)
                        except KeyError:
                            pass

            for np_idx, row in enumerate(new_fd_x):
                new_state = (row[0], row[1], row[2], row[3], row[4])
                new_action = (row[5], row[6], row[7])

                if expert == 1 and action == new_action:
                    if utils.states_are_close(new_state, state):
                        try:
                            self.vd_xs = self.vd_xs.drop(df_idx)
                            self.vd_ys = self.vd_ys.drop(df_idx)
                        except KeyError:
                            pass"""

        # step 2: create `tf.data.Dataset` from updated dataset
        # step 3: call `nn.sef_fit/validation_dataset()`
        utils.error('nothing found')

    @staticmethod
    def load_single_dataset(dataset_path: str) -> tf.data.Dataset:
        d = pd.read_csv(dataset_path)

        d_x = d.copy()
        d_y = d_x[['next_co2', 'next_voc', 'next_temp_in']].copy()
        d_x = d_x.drop(['next_co2', 'next_voc', 'next_temp_in'], axis=1)

        if utils.tf_dataset:
            dataset = tf.data.Dataset.from_tensor_slices((d_x.values.tolist(), d_y.values.tolist()))
            dataset = dataset.shuffle(buffer_size=10000, seed=utils.tf_seed, reshuffle_each_iteration=True)
        else:
            # Legacy
            dataset = (d_x.to_numpy(), d_y.to_numpy())

        return dataset

    @staticmethod
    def initialize_csv_datasets(timestamp: str):
        header = utils.dataset_header

        if not os.path.exists(utils.deploy_folder):
            os.makedirs(utils.deploy_folder)

        with open(utils.deploy_folder + 'fd_%s.csv' % timestamp, 'w') as f:
            f.write(header)
        with open(utils.deploy_folder + 'vd_%s.csv' % timestamp, 'w') as f:
            f.write(header)

    def set_fit_data(self, fd: tf.data.Dataset):
        self.fit_data = fd

    def set_validation_data(self, vd: tf.data.Dataset):
        self.validation_data = vd

    def set_test_data(self, td: tf.data.Dataset):
        self.test_data = td

    def fit(self, epochs: int, batch_size: int, fit_verbose: int = 2, save_best: bool = True, filepath: str = None):
        if self.fit_data is None:
            utils.error('(NN) Provide `fit_data`.')

        if save_best and filepath is None:
            utils.error('(NN) If `save_best` is set to True, you need to provide a valid `filepath`.')

        callbacks = []

        class ClearSessionCallBack(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                gc.collect()
                tf.keras.backend.clear_session()

        clear_session_callback = ClearSessionCallBack()
        callbacks.append(clear_session_callback)

        if save_best:
            verbose = 1 if utils.verbose else 0
            save_best_callback = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                                    monitor='val_loss',
                                                                    verbose=verbose,
                                                                    save_best_only=True)
            callbacks.append(save_best_callback)

        return self.model.fit(x=self.fit_data[0],
                              y=self.fit_data[1],
                              validation_data=self.validation_data,
                              epochs=epochs,
                              batch_size=batch_size,
                              verbose=fit_verbose,
                              callbacks=callbacks)

    def predict(self, td_x: dict = None, verbose: int = 2):
        if td_x is None:
            if self.test_data is None:
                utils.error('(NN) Both `td_x` and `test_data` are None. Provide at least one of them.')
            td_x = self.test_data[0]
        return self.model.predict(td_x, verbose=verbose)

    def save_model(self, dest: str):
        self.model.save(dest)


def reset_weights(model: tf.keras.Model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            reset_weights(layer)
            continue
        for k, initializer in layer.__dict__.items():
            if "initializer" not in k:
                continue
            # find the corresponding variable
            var = getattr(layer, k.replace("_initializer", ""))
            var.assign(initializer(var.shape, var.dtype))


def pypy_pickle(nn: NN, model_name: str):
    model = nn.model

    length = len(model.layers)
    weights = [model.layers[i].get_weights() for i in range(1, length, 1)]

    w_py = []
    for i in range(length - 1):
        w_py.append((weights[i][0].tolist(), weights[i][1].tolist()))

    nn = NNPyPy(w_py, nn.is_normalized)
    folder = utils.nn_folder + 'pickle/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    filepath = folder + model_name + '.pickle'
    with open(filepath, 'wb') as f:
        pickle.dump(nn, f)

    return filepath


def polyak_average(target_nn: NN, previous_nn: NN):
    if target_nn.layers != previous_nn.layers or target_nn.is_normalized != previous_nn.is_normalized:
        utils.error('(NN) Internal error! `target_nn` and `previous_nn` are different.')

    target_model = target_nn.model
    previous_model = previous_nn.model

    w_target = target_model.get_weights()
    w_previous = previous_model.get_weights()

    new_weights = list()

    for i in range(len(w_target)):
        if target_nn.is_normalized:
            if i == 0 or i == 1 or i == 2:
                new_weights.append(w_target[i])
                continue
        # Polyak exponentially decaying average
        update = (1 - utils.polyak_tau) * w_target[i] + utils.polyak_tau * w_previous[i]
        new_weights.append(update)

    target_model.set_weights(new_weights)
    previous_model.set_weights(new_weights)


def get_current_memory_usage():
    with open('/proc/self/status') as f:
        memusage = f.read().split('VmRSS:')[1].split('\n')[0][:-3]

    return int(memusage.strip()) / 1000
