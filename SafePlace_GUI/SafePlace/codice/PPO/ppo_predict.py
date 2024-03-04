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
# Ottieni il percorso assoluto della directory corrente
current_directory = os.path.dirname(os.path.abspath(__file__))

# Aggiungi la directory alla sys.path per permettere le importazioni relative
import sys
sys.path.append(current_directory)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["WANDB_START_METHOD"] = "thread"
from env_files.model_parameters import ModelParams
import env_files.utils as utils
from env_files.safeplace_env import SafePlaceEnv
from env_files.dataset_generation import *
import env_files.stats as stats 
from PPO_keras import PPO

import tensorflow as tf
import numpy as np

#import ok message
print("ppo_predict.py imports ok")

class PPO_predict:
    def __init__(self):
        env = SafePlaceEnv()
        params = ModelParams(exp_const=None, max_depth=None, rollout_moves=None,iterations=None)
        obs_space = 5
        action_dim = len(list(ActionData))

        seed_nn = 2020
        seed_temperature = 1243
        random.seed(seed_temperature)
        self.ppo_agent = PPO(seed_nn, obs_space, action_dim)

        name_model_actor = 'SafePlace/codice/PPO/models_pretrain/pretrain_SafePlaceEnv_keras_actor_seed2020_success107.5'
        self.ppo_agent.pi = tf.keras.models.load_model(f"{name_model_actor}.h5", compile=False)

    def get_predict_action(self, state):
        s = self.normalize_state(state.s[2:])
        s_action, s_logp, s_vf = self.ppo_agent.get_action(s)
        action_data = ActionData.get_action(s_action[0])

        return action_data

    def normalize_state(self, state):
        values = {"people": [0, 50],
        "co2": [400, 2500],
        "voc": [30, 1500],
        "temp_in": [-5, 40],
        "temp_out": [-5, 40]}
    
        state_normalized = []
        for i in range(len(state)):
            # values[list(values.keys())[i]][0] is the min value available for the feature that we are considering
            # in the first case we'll have: values['people'][0] = 0, values['people'][1] = 50 and so on...    
            s_norm = 2*((state[i] - values[list(values.keys())[i]][0]) / (values[list(values.keys())[i]][1] - values[list(values.keys())[i]][0])) - 1
            state_normalized.append(s_norm)

        return np.array([state_normalized])



'''
state = utils.initial_state()
s = normalize_state(state.s[2:])
s_action, s_logp, s_vf = ppo_agent.get_action(s)
action_data = ActionData.get_action(s_action[0])

next_state, rewards = env.do_transition(state, action_data)

state = next_state
'''







'''
def train():

    env = SafePlaceEnv()
    params = ModelParams(exp_const=None, max_depth=None, rollout_moves=None,iterations=None)

    obs_space = 5
    action_dim = len(list(ActionData))

    seed_temperature = 1243

    # take all csv file names
    csv_files = {}
    for root, dirs, files in os.walk('datasets/generated_reservations_profiles'):
        if len(dirs) == 0:
            key = root.split('/')[-1]
            csv_files[key] = sorted(files)

    csv_files = dict(sorted(csv_files.items()))

    
    ###################### create log file ######################
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/SafePlace_Pretrain/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_cum_Reward = log_dir + '/PPO_SafePlacetrain_log_' + str(run_num) + ".csv"
    log_stats = log_dir + '/PPO_SafePlaceTrain_stats_log_' + str(run_num) + ".csv"

    # logging file
    log_f = open(log_cum_Reward,"w+")
    log_f.write('profile, day, cumulative_reward\n')

    # logging file
    log_s = open(log_stats,"w+")
    log_s.write('profile, day, state, action, reward\n')
   
    
    ################################## 
    ############ TRAINING ############
    ################################## 
     
    # initialize a PPO agent
    seed_nn = 2020
    random.seed(seed_temperature)
    ppo_agent = PPO(seed_nn, obs_space, action_dim)

    # 1 if you want to use a "new" agent for each new environment, 2 otherwise
    var_training = 2
    # if you want to use a pretrained agent
    pretrain = True

    print('Var training: ', var_training)
    print('Pretrain: ', pretrain)

    if pretrain:
        # loading pretrained model
        name_model_actor = 'models_pretrain/pretrain_SafePlaceEnv_keras_actor_seed2020_success107.5'
        ppo_agent.pi = tf.keras.models.load_model(f"{name_model_actor}.h5")
        name_model_critic = 'models_pretrain/pretrain_SafePlaceEnv_keras_critic_seed2020_success107.5'
        ppo_agent.vf = tf.keras.models.load_model(f"{name_model_critic}.h5")
 
    for id_profile, profile_files in csv_files.items():
        # each profile contains more days
        mean_reward = []

        if var_training == 1:
            # get new agent
            ppo_agent = PPO(seed_nn, obs_space, action_dim)

            if pretrain:
                # loading pretrained model
                name_model_actor = 'pretrain_SafePlaceEnv_keras_actor_seed2020_success107.5'
                ppo_agent.pi = tf.keras.models.load_model(f"{name_model_actor}.h5")
                name_model_critic = 'pretrain_SafePlaceEnv_keras_critic_seed2020_success107.5'
                ppo_agent.vf = tf.keras.models.load_model(f"{name_model_critic}.h5")

        for day, file in enumerate(profile_files):
            stats_list = []

            # each file is equivalent to one day
            utils.initialize_all(params, env)
            utils.update_reservations(reservations_filepath=f'datasets/generated_reservations_profiles/{id_profile}/{file}', random_initial_temp_in=True)

            # reset the state
            state = utils.initial_state()
            cum_reward = []

            while not state.is_terminal:
                if utils.step_stats:
                    s_stats_time = stats.step_stats_start()
                
                s = normalize_state(state.s[2:])
                s_action, s_logp, s_vf = ppo_agent.get_action(s)
                action_data = ActionData.get_action(s_action[0])

                next_state, rewards = env.do_transition(state, action_data)

                if utils.step_stats:
                    stats.step_stats_record_data(state, ActionData.get_action(s_action[0]), rewards, s_stats_time, 1, 0)

                stats_list.append({'State': state,
                            'Action':  ActionData.get_action(s_action[0]),
                            'Reward': rewards})

                state = next_state
                cum_reward.append(rewards[0])
                ppo_agent.buffer.store(
                        s, 
                        s_action[0], 
                        s_logp[0], 
                        rewards[0], 
                        s_vf.squeeze()
                    )

            l_state = s
            l_vf = ppo_agent.vf(l_state)
            ppo_agent.buffer.compute_mc(l_vf, 120)
            ppo_agent.update()   
            
            print_cum_reward = round(sum(cum_reward),2)
            mean_reward.append(print_cum_reward)

            print("Profile: {}\t Day: {}\t Cumulative reward: {}".format(id_profile, day, print_cum_reward))
            log_f.write('{},{},{}\n'.format(id_profile, day, print_cum_reward))
            log_f.flush()

            for stat in stats_list:
                log_s.write('{},{},{},{},{}\n'.format(id_profile, day, stat['State'], stat['Action'], stat['Reward']))
            log_s.flush()
            
        
        ppo_agent.pi.save(f'models_train/SafePlaceEnv_keras_actor_seed{seed_nn}_Profile{id_profile}_meanCumulativeReward_{np.mean(np.array(mean_reward))}.h5')
        ppo_agent.vf.save(
            f'models_train/SafePlaceEnv_keras_critic_seed{seed_nn}_Profile{id_profile}_meanCumulativeReward_{np.mean(np.array(mean_reward))}.h5')
        
       
if __name__ == '__main__':
    train()
    
'''  