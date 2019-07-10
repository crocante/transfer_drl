import sys
sys.path.append('/data/GuoJiajia/PycharmProjects/transfer_drl/')
import gym
import tensorflow as tf
import os
import yaml
import numpy as np
from dpagent.dyn_model.dynamics_model import Dyn_Model
from dpagent.dyn_model.policy_random import Policy_Random
from dpagent.dyn_model.collect_samples import CollectSamples
from dpagent.dyn_model.data_manipulation import generate_training_data_inputs
from dpagent.dyn_model.data_manipulation import generate_training_data_outputs

def perform_rollouts(policy, num_rollouts, steps_per_rollout, CollectSamples, env):
    #collect training data by performing rollouts
    print("Beginning to do ", num_rollouts, " rollouts.")
    c = CollectSamples(env, policy)
    states, controls, rewards_list, rolloutrewards_list = c.collect_samples(num_rollouts, steps_per_rollout)

    print("Performed ", len(states), " rollouts, each with ", states[0].shape[0], " steps.")
    return states, controls, rewards_list, rolloutrewards_list

def perform_step(state, action, policy, CollectSamples, env):
    c = CollectSamples(env, policy)
    states, rewards, terminal = c.perform_step(state, action)
    return states, rewards, terminal

def normalize(data, mean, std):
    data = data - mean
    data = np.nan_to_num(data / std)
    return data

if __name__ == '__main__':
    yaml_path = os.path.abspath('../../yaml/halfcheetah.yaml')
    assert (os.path.exists(yaml_path))
    with open(yaml_path, 'r') as f:
        params = yaml.load(f)
    # data collection
    num_rollouts_train = params['data_collection']['num_rollouts_train']
    num_rollouts_val = params['data_collection']['num_rollouts_val']
    #dynamics
    num_fc_layers = params['dyn_model']['num_fc_layers']
    depth_fc_layers = params['dyn_model']['depth_fc_layers']
    batchsize = params['dyn_model']['batchsize']
    lr = params['dyn_model']['lr']
    nEpoch = params['dyn_model']['nEpoch']
    # steps
    dt_steps = params['steps']['dt_steps']
    steps_per_episode = params['steps']['steps_per_episode']
    steps_per_rollout_train = params['steps']['steps_per_rollout_train']
    steps_per_rollout_val = params['steps']['steps_per_rollout_val']
    # aggregation
    num_aggregation_iters = params['aggregation']['num_aggregation_iters']
    save_dir = 'run_0608'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir+'/losses')
        os.makedirs(save_dir+'/models')
        os.makedirs(save_dir+'/saved_forwardsim')
        os.makedirs(save_dir+'/saved_trajfollow')
        os.makedirs(save_dir+'/training_data')

    env = gym.make('HalfCheetah-v1')
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    random_policy = Policy_Random(env)

    gpu_device = 1
    gpu_frac = 0.3
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
    config = tf.ConfigProto(gpu_options=gpu_options,
                            log_device_placement=False,
                            allow_soft_placement=True,
                            inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1)

    with tf.Session(config=config) as sess:
        # perform rollouts
        states, controls, rewards, rollout_rewards = perform_rollouts(random_policy, num_rollouts_train, steps_per_rollout_train,
                                                  CollectSamples, env)
        states_val, controls_val, rewards_val, rollout_rewards_val = perform_rollouts(random_policy, num_rollouts_val, steps_per_rollout_val,
                                                          CollectSamples, env)
        dataX, dataY = generate_training_data_inputs(states, controls)
        dataZ, reward = generate_training_data_outputs(states, rewards)
        dataX_val, dataY_val = generate_training_data_inputs(states_val, controls_val)
        dataZ_val, reward_val = generate_training_data_outputs(states_val, rewards_val)

        # errors_1_per_agg = []
        # errors_5_per_agg = []
        # errors_10_per_agg = []
        # errors_50_per_agg = []
        # errors_100_per_agg = []

        mean_x = np.mean(dataX, axis=0)
        dataX = dataX - mean_x
        std_x = np.std(dataX, axis=0)
        dataX = np.nan_to_num(dataX / std_x)
        dataX_val = normalize(dataX_val, mean_x, std_x)

        mean_y = np.mean(dataY, axis=0)
        dataY = dataY - mean_y
        std_y = np.std(dataY, axis=0)
        dataY = np.nan_to_num(dataY / std_y)
        dataY_val = normalize(dataY_val, mean_y, std_y)

        mean_z = np.mean(dataZ, axis=0)
        dataZ = dataZ - mean_z
        std_z = np.std(dataZ, axis=0)
        dataZ = np.nan_to_num(dataZ / std_z)
        dataZ_val = normalize(dataZ_val, mean_z, std_z)

        mean_r = np.mean(reward, axis=0)
        reward = reward - mean_r
        std_r = np.std(reward, axis=0)
        reward = np.nan_to_num(reward / std_r)
        reward_val = normalize(reward_val, mean_r, std_r)

        ## concatenate state and action, to be used for training dynamics
        inputs = np.concatenate((dataX, dataY), axis=1)
        inputs_val = np.concatenate((dataX_val, dataY_val), axis=1)
        # reward = np.array([reward].T)
        reward = reward.reshape(-1, 1)
        reward_val = reward_val.reshape(-1, 1)
        outputs = np.concatenate((dataZ, reward), axis=1)
        outputs_val = np.concatenate((dataZ_val, reward_val), axis=1)
        assert inputs.shape[0] == outputs.shape[0]
        assert inputs_val.shape[0] == outputs_val.shape[0]
        inputSize = inputs.shape[1]
        outputSize = outputs.shape[1]
        dyn_model = Dyn_Model(inputSize, outputSize, sess, lr, batchsize, num_fc_layers, depth_fc_layers,
                              mean_x, mean_y, mean_z, mean_r, std_x, std_y, std_z, std_r)

        sess.run(tf.global_variables_initializer())
        counter_agg_iters = 0
        training_loss_list = []
        while (counter_agg_iters < num_aggregation_iters):
            training_loss = dyn_model.train(inputs, outputs, nEpoch, save_dir)
            training_loss_list.append(training_loss)
            print("\nTraining loss: ", training_loss)
            counter_agg_iters = counter_agg_iters + 1

            validation_loss = dyn_model.run_validation(inputs_val, outputs_val)

            # iters = 0
            # prediction_steps = []
            # prediction_rewards = []
            # lable_states = []
            # lable_rewards = []
            # step_losses = []
            # reward_losses = []
            # while (iters<num_rollouts_val):
            #     step = 0
            #     state = env.reset()
            #     terminal = False
            #     while(not terminal):
            #         action = random_policy.get_action(state)
            #
            #         next_state, reward, terminal = perform_step(state, action[0], random_policy, CollectSamples, env)
            #         next_state = normalize(next_state, mean_z, std_z)
            #         reward = normalize(reward, mean_r, std_r)
            #
            #         prediction_step, prediction_reward = dyn_model.do_forward(state, action[0])
            #
            #         step_loss = np.sum(np.square(next_state-prediction_step),axis=1) / next_state.shape[0]
            #         reward_loss = np.square(reward-prediction_reward)
            #         # if(step%100==0):
            #         #     print("\nstep_loss:", step_loss[0])
            #         #     print("\nreward_loss:", reward_loss[0][0])
            #
            #         prediction_steps.append(prediction_step)
            #         prediction_rewards.append(prediction_reward)
            #         lable_states.append(next_state)
            #         lable_rewards.append(reward)
            #         step_losses.append(step_loss[0])
            #         reward_losses.append(reward_loss[0][0])
            #
            #         state = next_state
            #         step += 1
            #
            #     iters += 1
            #
            # np.save(save_dir + '/losses/prediction_steps_' + str(counter_agg_iters)+'.npy', prediction_steps)
            # np.save(save_dir + '/losses/prediction_rewards_' + str(counter_agg_iters)+'.npy', prediction_rewards)
            # np.save(save_dir + '/losses/lable_states_' + str(counter_agg_iters) + '.npy', lable_states)
            # np.save(save_dir + '/losses/lable_rewards_' + str(counter_agg_iters) + '.npy', lable_rewards)
            # np.savetxt(save_dir + '/losses/step_losses_' + str(counter_agg_iters) + '.txt', step_losses)
            # np.savetxt(save_dir + '/losses/reward_losses_' + str(counter_agg_iters) + '.txt', reward_losses)

            #long horizon prediction
            # validation_inputs_states = []
            # labels_1step = []
            # labels_5step = []
            # labels_10step = []
            # labels_50step = []
            # labels_100step = []
            # labels_1reward = []
            # labels_5reward = []
            # labels_10reward = []
            # labels_50reward = []
            # labels_100reward = []
            # controls_100step = []
            #
            # #####################################
            # ## make the arrays to pass into forward sim
            # #####################################
            #
            # for i in range(num_rollouts_val):
            #
            #     length_curr_rollout = states_val[i].shape[0]
            #
            #     if (length_curr_rollout > 100):
            #
            #         #########################
            #         #### STATE INPUTS TO NN
            #         #########################
            #
            #         ## take all except the last 100 pts from each rollout
            #         validation_inputs_states.append(states_val[i][0:length_curr_rollout - 100])
            #
            #         #########################
            #         #### CONTROL INPUTS TO NN
            #         #########################
            #
            #         # 100 step controls
            #         list_100 = []
            #         for j in range(100):
            #             list_100.append(controls_val[i][0 + j:length_curr_rollout - 100 + j])
            #             ##for states 0:x, first apply acs 0:x, then apply acs 1:x+1, then apply acs 2:x+2, etc...
            #         list_100 = np.array(list_100)  # 100xstepsx2
            #         list_100 = np.swapaxes(list_100, 0, 1)  # stepsx100x2
            #         controls_100step.append(list_100)
            #
            #         #########################
            #         #### STATE LABELS- compare these to the outputs of NN (forward sim)
            #         #########################
            #         labels_1step.append(states_val[i][0 + 1:length_curr_rollout - 100 + 1])
            #         labels_5step.append(states_val[i][0 + 5:length_curr_rollout - 100 + 5])
            #         labels_10step.append(states_val[i][0 + 10:length_curr_rollout - 100 + 10])
            #         labels_50step.append(states_val[i][0 + 50:length_curr_rollout - 100 + 50])
            #         labels_100step.append(states_val[i][0 + 100:length_curr_rollout - 100 + 100])
            #         labels_1reward.append(rewards_val[i][0 + 1:length_curr_rollout - 100 + 1])
            #
            # validation_inputs_states = np.concatenate(validation_inputs_states)
            # controls_100step = np.concatenate(controls_100step)
            # labels_1step = np.concatenate(labels_1step)
            # labels_5step = np.concatenate(labels_5step)
            # labels_10step = np.concatenate(labels_10step)
            # labels_50step = np.concatenate(labels_50step)
            # labels_100step = np.concatenate(labels_100step)
            #
            # many_in_parallel = True
            # predicted_100step = dyn_model.do_forward_sim(validation_inputs_states, controls_100step, many_in_parallel)
            #
            # array_meanx = np.tile(np.expand_dims(mean_x, axis=0), (labels_1step.shape[0], 1))
            # array_stdx = np.tile(np.expand_dims(std_x, axis=0), (labels_1step.shape[0], 1))
            #
            # error_1step = np.mean(np.square(np.nan_to_num(np.divide(predicted_100step[1] - array_meanx, array_stdx))
            #                                 - np.nan_to_num(np.divide(labels_1step - array_meanx, array_stdx))))
            # error_5step = np.mean(np.square(np.nan_to_num(np.divide(predicted_100step[5] - array_meanx, array_stdx))
            #                                 - np.nan_to_num(np.divide(labels_5step - array_meanx, array_stdx))))
            # error_10step = np.mean(np.square(np.nan_to_num(np.divide(predicted_100step[10] - array_meanx, array_stdx))
            #                                  - np.nan_to_num(np.divide(labels_10step - array_meanx, array_stdx))))
            # error_50step = np.mean(np.square(np.nan_to_num(np.divide(predicted_100step[50] - array_meanx, array_stdx))
            #                                  - np.nan_to_num(np.divide(labels_50step - array_meanx, array_stdx))))
            # error_100step = np.mean(np.square(np.nan_to_num(np.divide(predicted_100step[100] - array_meanx, array_stdx))
            #                                   - np.nan_to_num(np.divide(labels_100step - array_meanx, array_stdx))))
            # print("Multistep error values: ", error_1step, error_5step, error_10step, error_50step, error_100step, "\n")
            #
            # errors_1_per_agg.append(error_1step)
            # errors_5_per_agg.append(error_5step)
            # errors_10_per_agg.append(error_10step)
            # errors_50_per_agg.append(error_50step)
            # errors_100_per_agg.append(error_100step)
