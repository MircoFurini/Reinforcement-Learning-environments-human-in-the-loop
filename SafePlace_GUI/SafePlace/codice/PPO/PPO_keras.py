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

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from buffer_keras import Buffer
import numpy as np

class PPO:

    def __init__(self, seed, obs_space, action_dim):
    
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.pi = self.MLP(seed, self.obs_space, self.action_dim, actor=True)
        self.pi_opt = tf.keras.optimizers.Adam()
        self.pi_epochs = 10

        self.vf = self.MLP(seed, obs_space, action_dim)
        self.vf_opt = tf.keras.optimizers.Adam()
        self.vf_epochs = 10
        
        self.update_freq = 5
        b_capacity = 120
        self.buffer = Buffer(b_capacity, obs_size=obs_space)

        self.clip = 0.2
        self.entr_coeff = 1e-3
        self.target_kl = 0.01

        tf.random.set_seed(seed)
        np.random.seed(seed)


    def get_action(self, s_state):
        probs = self.pi(s_state).numpy()
        s_action = [np.random.choice(self.action_dim, p=prob) for prob in probs]
        s_logp = np.log([prob[action] for prob, action in zip(probs, s_action)])
        s_vf = self.vf(s_state).numpy()
        return s_action, s_logp, s_vf

   
    def MLP(self, seed, obs_space, action_dim, actor=False):
        tf.random.set_seed(seed)
        np.random.seed(seed)
        
        state_input = Input(shape=obs_space, name='input')
        h = state_input

        for i in range(2):
            h = Dense(64, activation='relu', name='hidden_' + str(i))(h)

        if actor:
            y = Dense(action_dim, activation='softmax', name='pi_output')(h)  
        else:
            y = Dense(1, activation='linear', name='vf_output')(h)
        
        model = Model(inputs=state_input, outputs=y)

        return model

    def update(self):
       
        u_pi_loss, u_entropy, u_vf_loss = 0, 0, 0
        b_state, b_action, b_logp, b_adv, b_ret = self.buffer.sample()

        b_state = tf.convert_to_tensor(b_state, dtype=tf.float32)
        b_action = tf.convert_to_tensor(b_action, dtype=tf.int32)
        b_logp = tf.convert_to_tensor(b_logp, dtype=tf.float32)
        b_adv = tf.convert_to_tensor(b_adv, dtype=tf.float32)
        b_ret = tf.convert_to_tensor(b_ret, dtype=tf.float32)
                    
        pi_loss, entropy = self.fit_pi(b_state, b_action, b_logp, b_adv)
        u_pi_loss += pi_loss
        u_entropy += entropy

        u_vf_loss += self.fit_vf(b_state, b_ret)

        self.buffer.clear()

    @tf.function(experimental_relax_shapes=True)
    def fit_pi(self, b_state, b_action, b_logp, b_adv):
        cum_entropy, cum_pi_loss = 0.0, 0.0

        for pi_e in tf.range(1, self.pi_epochs):
            with tf.GradientTape() as tape:            
                probs = self.pi(b_state)
                a_logprob = probs *  tf.one_hot(b_action, self.action_dim)

                a_logprob= tf.reduce_sum(a_logprob, axis=-1)

                a_logprob = tf.math.log(a_logprob)
            
                logratio = a_logprob - b_logp
                ratio = tf.exp(logratio)

                clip_adv = tf.where(b_adv > 0, (1 + self.clip) * b_adv, (1 - self.clip) * b_adv)
                surr_adv = tf.reduce_mean(tf.minimum(ratio * b_adv, clip_adv))
                entropy = tf.reduce_mean(-tf.reduce_sum(probs * tf.math.log(probs), axis=-1)) * self.entr_coeff

                pi_loss = -surr_adv + entropy

                pi_grad = tape.gradient(pi_loss, self.pi.trainable_variables)
                self.pi_opt.apply_gradients(zip(pi_grad, self.pi.trainable_variables))

            cum_entropy += entropy
            cum_pi_loss += pi_loss

            
            approx_kl = tf.reduce_mean(tf.math.expm1(logratio) - logratio)
            if approx_kl > self.target_kl:
                #tf.print(f"Early stop at epoch due to reaching max kl")
                break

        pi_e = tf.cast(pi_e, dtype=tf.float32)
        return cum_pi_loss / pi_e, cum_entropy / pi_e

    @tf.function(experimental_relax_shapes=True)
    def fit_vf(self, b_state, b_ret):  
        cum_vf_loss = 0.0

        for vf_e in tf.range(self.vf_epochs):

            with tf.GradientTape() as tape:
                v = tf.squeeze(self.vf(b_state))
                vf_loss = tf.reduce_mean(tf.square((b_ret - v)))
            
            v_grad = tape.gradient(vf_loss, self.vf.trainable_variables)
            self.vf_opt.apply_gradients(zip(v_grad, self.vf.trainable_variables))
        
            cum_vf_loss += vf_loss

        return cum_vf_loss / tf.cast(vf_e, dtype=tf.float32)

     
       