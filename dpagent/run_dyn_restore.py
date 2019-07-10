import os
import tensorflow as tf
import yaml
import time
import numpy as np
from collections import OrderedDict
from envs.half_cheetah_rand_direc import HalfCheetahRandDirecEnv
from dpagent.dyn_model.data_manipulation import generate_training_data_inputs, generate_training_data_inputs_env
from dpagent.dyn_model.data_manipulation import generate_training_data_outputs, generate_training_data_outputs_env
from envs.normalized_env import normalize
from tt_utils.logx import EpochLogger
from tt_utils.mpi_tools import num_procs
from dpagent.agent_adapt.dp_adapt import DP_Adapt
import dpagent.agent_adapt.core as core
from dpagent.agent_adapt.sampler import collect_samples, collect_samples_comp, add_noise, get_model_samples, get_samples, get_samples_with_buf

def get_mean_std(dataX, dataY, dataZ, reward, agent):
    mean_x = np.mean(dataX, axis=0)
    dataX = dataX - mean_x
    std_x = np.std(dataX, axis=0)
    dataX = np.nan_to_num(dataX / std_x)
    agent.mean_x = mean_x
    agent.std_x = std_x

    mean_y = np.mean(dataY, axis=0)
    dataY = dataY - mean_y
    std_y = np.std(dataY, axis=0)
    dataY = np.nan_to_num(dataY / std_y)
    agent.mean_y = mean_y
    agent.std_y = std_y

    mean_z = np.mean(dataZ, axis=0)
    dataZ = dataZ - mean_z
    std_z = np.std(dataZ, axis=0)
    dataZ = np.nan_to_num(dataZ / std_z)
    agent.mean_z = mean_z
    agent.std_z = std_z

    mean_r = np.mean(reward, axis=0)
    reward = reward - mean_r
    std_r = np.std(reward, axis=0)
    reward = np.nan_to_num(reward / std_r)
    agent.mean_r = mean_r
    agent.std_r = std_r
    return dataX, mean_x, std_x, dataY, mean_y, std_y, dataZ, mean_z, std_z, reward, mean_r, std_r

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file', type=str, default='halfcheetah_rand_dirc')
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    yaml_path = os.path.abspath('../yaml/' + args.yaml_file + '.yaml')
    assert (os.path.exists(yaml_path))
    with open(yaml_path, 'r') as f:
        params = yaml.load(f)
    # ppo
    epochs = args.epochs
    steps_per_epoch = args.steps
    gamma = args.gamma
    seed = args.seed
    max_ep_len = 1000
    save_freq = 10
    train_pi_iters = 50
    train_v_iters = 50
    target_kl = 0.01
    # dyn_model
    num_fc_layers = params['dyn_model']['num_fc_layers']
    depth_fc_layers = params['dyn_model']['depth_fc_layers']
    batchsize = params['dyn_model']['batchsize']
    lr = params['dyn_model']['lr']
    nEpoch = params['dyn_model']['nEpoch']
    fraction_use_new = params['dyn_model']['fraction_use_new']
    # aggregation
    num_aggregation_iters = params['aggregation']['num_aggregation_iters']
    num_trajectories_for_aggregation = params['aggregation']['num_trajectories_for_aggregation']
    rollouts_forTraining = params['aggregation']['rollouts_forTraining']
    inner_aggregation_iters_k = params['aggregation']['inner_aggregation_iters_k']
    # noise
    make_aggregated_dataset_noisy = params['noise']['make_aggregated_dataset_noisy']
    # data collection
    use_threading = params['data_collection']['use_threading']
    num_rollouts_train = params['data_collection']['num_rollouts_train']
    num_rollouts_val = params['data_collection']['num_rollouts_val']
    # steps
    dt_steps = params['steps']['dt_steps']
    steps_per_episode = params['steps']['steps_per_episode']
    steps_per_rollout_train = params['steps']['steps_per_rollout_train']
    steps_per_rollout_val = params['steps']['steps_per_rollout_val']
    noiseToSignal = 0.01
    from tt_utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    save_dir = 'data/run_0702/restore_dyn'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir + '/losses')
        os.makedirs(save_dir + '/models')
        os.makedirs(save_dir + '/training_data')

    env = HalfCheetahRandDirecEnv()
    env = normalize(env)    # apply normalize wrapper to env

    inputSize = env.observation_space.shape[0] + env.action_space.shape[0]
    outputSize = env.observation_space.shape[0] + 1
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    # model_logger = EpochLogger(**logger_kwargs)
    logger = EpochLogger(**logger_kwargs)

    gpu_device = 4
    gpu_frac = 0.3
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
    config = tf.ConfigProto(gpu_options=gpu_options,
                            log_device_placement=False,
                            allow_soft_placement=True,
                            inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1)

    with tf.Session(config=config) as sess:
        saver = tf.train.import_meta_graph('data/run_0702/origin/models/800.meta')
        saver.restore(sess, tf.train.latest_checkpoint('data/run_0702/origin/models'))

        # graph_ = tf.get_default_graph().get_all_collection_keys()
        params = tf.get_default_graph().get_collection("trainable_variables")

        odict = OrderedDict()
        for var in params:
            odict[var.name] = var
        agent = DP_Adapt(odict, inputSize, outputSize, env, local_steps_per_epoch, lr, actor_critic=core.mlp_actor_critic)
        agent.initialize(sess)

        pi_loss_list = []
        v_loss_list = []
        dyn_loss_list = []
        ep_return = []
        training_loss_list = []
        old_loss_list = []
        new_loss_list = []
        errors_1_per_agg = []
        errors_5_per_agg = []
        errors_10_per_agg = []
        errors_50_per_agg = []
        errors_100_per_agg = []
        # o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        counter_agg_iters = 0
        observations, actions, rewards = collect_samples(num_rollouts_train, env, steps_per_rollout_train,
                                                         sess, agent, logger, max_ep_len)

        observations_val, actions_val, rewards_val = collect_samples(num_rollouts_val, env, steps_per_rollout_val,
                                                         sess, agent, logger, max_ep_len)
        # ep_return.append(ep_ret)
        # observations = np.array(observations)
        # actions = np.array(actions)

        dataX, dataY = generate_training_data_inputs(observations, actions)
        dataZ, reward = generate_training_data_outputs(observations, rewards)
        dataX_new = np.zeros((0, dataX.shape[1]))
        dataY_new = np.zeros((0, dataY.shape[1]))
        dataZ_new = np.zeros((0, dataZ.shape[1]))
        reward_new = np.zeros((0,))
        dataX, mean_x, std_x, dataY, mean_y, std_y, dataZ, mean_z, std_z, reward, mean_r, std_r = \
            get_mean_std(dataX, dataY, dataZ, reward, agent)

        inputs = np.concatenate((dataX, dataY), axis=1)
        reward = reward.reshape(-1, 1)
        outputs = np.concatenate((dataZ, reward), axis=1)
        assert inputs.shape[0] == outputs.shape[0]
        # start_time = time.time()
        x_list = []
        x_model_list = []
        z_list = []
        z_model_list = []
        rewards_list = []
        rewards_model_list = []
        return_list = []
        return_model_list = []
        while (counter_agg_iters < num_aggregation_iters+1):
            validation_inputs_states = []
            labels_1step = []
            labels_5step = []
            labels_10step = []
            labels_50step = []
            labels_100step = []
            controls_100step = []

            #####################################
            ## make the arrays to pass into forward sim
            #####################################

            for i in range(num_rollouts_val):

                length_curr_rollout = observations_val[i].shape[0]

                if (length_curr_rollout > 100):

                    #########################
                    #### STATE INPUTS TO NN
                    #########################

                    ## take all except the last 100 pts from each rollout
                    validation_inputs_states.append(observations_val[i][0:length_curr_rollout - 100])

                    #########################
                    #### CONTROL INPUTS TO NN
                    #########################

                    # 100 step controls
                    list_100 = []
                    for j in range(100):
                        list_100.append(actions_val[i][0 + j:length_curr_rollout - 100 + j])
                        ##for states 0:x, first apply acs 0:x, then apply acs 1:x+1, then apply acs 2:x+2, etc...
                    list_100 = np.array(list_100)  # 100xstepsx2
                    list_100 = np.swapaxes(list_100, 0, 1)  # stepsx100x2
                    controls_100step.append(list_100)

                    #########################
                    #### STATE LABELS- compare these to the outputs of NN (forward sim)
                    #########################
                    labels_1step.append(observations_val[i][0 + 1:length_curr_rollout - 100 + 1])
                    labels_5step.append(observations_val[i][0 + 5:length_curr_rollout - 100 + 5])
                    labels_10step.append(observations_val[i][0 + 10:length_curr_rollout - 100 + 10])
                    labels_50step.append(observations_val[i][0 + 50:length_curr_rollout - 100 + 50])
                    labels_100step.append(observations_val[i][0 + 100:length_curr_rollout - 100 + 100])

            validation_inputs_states = np.concatenate(validation_inputs_states)
            controls_100step = np.concatenate(controls_100step)
            labels_1step = np.concatenate(labels_1step)
            labels_5step = np.concatenate(labels_5step)
            labels_10step = np.concatenate(labels_10step)
            labels_50step = np.concatenate(labels_50step)
            labels_100step = np.concatenate(labels_100step)

            #####################################
            ## pass into forward sim, to make predictions
            #####################################

            predicted_100step = agent.do_forward_sim(validation_inputs_states, controls_100step)

            #####################################
            ## Calculate validation metrics (mse loss between predicted and true)
            #####################################

            array_meanx = np.tile(np.expand_dims(mean_x, axis=0), (labels_1step.shape[0], 1))
            array_stdx = np.tile(np.expand_dims(std_x, axis=0), (labels_1step.shape[0], 1))

            error_1step = np.mean(np.square(np.nan_to_num(np.divide(predicted_100step[1] - array_meanx, array_stdx))
                                            - np.nan_to_num(np.divide(labels_1step - array_meanx, array_stdx))))
            error_5step = np.mean(np.square(np.nan_to_num(np.divide(predicted_100step[5] - array_meanx, array_stdx))
                                            - np.nan_to_num(np.divide(labels_5step - array_meanx, array_stdx))))
            error_10step = np.mean(np.square(np.nan_to_num(np.divide(predicted_100step[10] - array_meanx, array_stdx))
                                             - np.nan_to_num(np.divide(labels_10step - array_meanx, array_stdx))))
            error_50step = np.mean(np.square(np.nan_to_num(np.divide(predicted_100step[50] - array_meanx, array_stdx))
                                             - np.nan_to_num(np.divide(labels_50step - array_meanx, array_stdx))))
            error_100step = np.mean(np.square(np.nan_to_num(np.divide(predicted_100step[100] - array_meanx, array_stdx))
                                              - np.nan_to_num(np.divide(labels_100step - array_meanx, array_stdx))))
            print("current iteration:", counter_agg_iters, "\n")
            print("Multistep error values: ", error_1step, error_5step, error_10step, error_50step, error_100step, "\n")

            errors_1_per_agg.append(error_1step)
            errors_5_per_agg.append(error_5step)
            errors_10_per_agg.append(error_10step)
            errors_50_per_agg.append(error_50step)
            errors_100_per_agg.append(error_100step)

            dataX_new_preprocessed = np.nan_to_num((dataX_new - mean_x) / std_x)
            dataY_new_preprocessed = np.nan_to_num((dataY_new - mean_y) / std_y)
            dataZ_new_preprocessed = np.nan_to_num((dataZ_new - mean_z) / std_z)
            reward_new_preprocessed = np.nan_to_num((reward_new - mean_r) / std_r)

            # concatenate state and action, to be used for training dynamics
            inputs_new = np.concatenate((dataX_new_preprocessed, dataY_new_preprocessed), axis=1)
            # reward_new_preprocessed = reward_new_preprocessed.reshape(-1, 1)
            outputs_new = np.concatenate((dataZ_new_preprocessed, reward_new_preprocessed.reshape(-1, 1)), axis=1)

            print("current iteration:", counter_agg_iters, "\n")
            training_loss, old_loss, new_loss = agent.train(inputs, outputs, inputs_new, outputs_new, batchsize,
                                                                nEpoch, save_dir, fraction_use_new)
            # how good is model on training data
            training_loss_list.append(training_loss)
            # how good is model on old dataset
            old_loss_list.append(old_loss)
            # how good is model on new dataset
            new_loss_list.append(new_loss)

            ###################dagger########################
            list_rewards = []
            starting_states = []
            selected_multiple_u = []
            resulting_multiple_x = []
            for rollout_num in range(num_trajectories_for_aggregation):
                ##########Performing MPC rollout
                # starting_observation, starting_state = env.reset()
                starting_observation, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
                #start swimmer heading in correct direction
                #desired trajectory to follow
                # starting_observation_NNinput = from_observation_to_usablestate(starting_observation, which_agent, True)

                x, x_m, z, z_m, selected_u, ep_rew, ep_rew_model, ret, ret_model, resulting_x = collect_samples_comp(1, env,
                                                             steps_per_rollout_train, sess, agent, logger, max_ep_len)
                list_rewards.append(ep_rew)
                selected_multiple_u.append(selected_u[0])
                resulting_multiple_x.append(resulting_x[0])
                for i in range(steps_per_rollout_train):
                    x_list.append(x[i])
                    z_list.append(z[i])
                    x_model_list.append(x_m[i])
                    z_model_list.append(z_m[i])
                    rewards_list.append(ep_rew[i])
                    rewards_model_list.append(ep_rew_model[i])
                return_list.append(ret[0])
                return_model_list.append(ret_model[0])

            if (counter_agg_iters < (num_aggregation_iters - 1)):

                ##############################
                ### aggregate some rollouts into training set
                ##############################

                x_array = np.array(resulting_multiple_x)[0:(rollouts_forTraining + 1)]
                # uu = np.array(selected_multiple_u)
                # u_array = np.squeeze(np.array(selected_multiple_u), axis=2)[0:(rollouts_forTraining + 1)]
                u_array = np.array(selected_multiple_u)[0:(rollouts_forTraining + 1)]
                r_array = np.array(list_rewards)[0:(rollouts_forTraining + 1)]

                for i in range(rollouts_forTraining):

                    x = x_array[i]  # [N+1, NN_inp]
                    u = u_array[i]  # [N, actionSize]
                    r = np.squeeze(r_array[i].reshape(-1, 1), axis=1)   #[N,]

                    newDataX = np.copy(x[0:-1, :])
                    newDataY = np.copy(u)
                    newDataZ = np.copy(x[1:, :] - x[0:-1, :])
                    newReward = np.copy(r)

                    # make this new data a bit noisy before adding it into the dataset
                    if (make_aggregated_dataset_noisy):
                        newDataX = add_noise(newDataX, noiseToSignal)
                        newDataZ = add_noise(newDataZ, noiseToSignal)

                    # the actual aggregation
                    dataX_new = np.concatenate((dataX_new, newDataX))
                    dataY_new = np.concatenate((dataY_new, newDataY))
                    dataZ_new = np.concatenate((dataZ_new, newDataZ))
                    reward_new = np.concatenate((reward_new, newReward))

                ##############################
                ### aggregate the rest of the rollouts into validation set
                ##############################

                x_array = np.array(resulting_multiple_x)[rollouts_forTraining:len(resulting_multiple_x)]
                # ^ dim: [rollouts_forValidation x stepsPerEpisode+1 x stateSize]
                # u_array = np.squeeze(np.array(selected_multiple_u), axis=2)[
                #               rollouts_forTraining:len(resulting_multiple_x)]
                u_array = np.array(selected_multiple_u)[rollouts_forTraining:len(resulting_multiple_x)]
                    # rollouts_forValidation x stepsPerEpisode x acSize
                r_array = np.array(list_rewards)[rollouts_forTraining:len(resulting_multiple_x)]
                full_states_list = []
                full_controls_list = []
                # for i in range(len(observations_val)):
                #     full_states_list.append(observations_val[i])
                #     full_controls_list.append(actions_val[i])
                for i in range(x_array.shape[0]):
                    x = np.array(x_array[i])
                    observations_val.append(x[0:-1, :])
                    actions_val.append(np.squeeze(u_array[i]))
                # states_val = np.array(full_states_list)
                # controls_val = np.array(full_controls_list)
            # if(counter_agg_iters%20==0):
            #     np.savetxt(save_dir + '/training_data/env_x_' + str(counter_agg_iters)+'.txt', x_list)
            #     np.savetxt(save_dir + '/training_data/env_z_' + str(counter_agg_iters)+'.txt', z_list)
            #     np.savetxt(save_dir + '/training_data/model_x_' + str(counter_agg_iters)+'.txt', x_model_list)
            #     np.savetxt(save_dir + '/training_data/model_z_' + str(counter_agg_iters)+'.txt', z_model_list)
            #     np.savetxt(save_dir + '/training_data/env_rewards_' + str(counter_agg_iters)+'.txt', rewards_list)
            #     np.savetxt(save_dir + '/training_data/model_rewards_' + str(counter_agg_iters)+'.txt', rewards_model_list)

            counter_agg_iters = counter_agg_iters + 1
        # np.savetxt(save_dir + '/training_data/env_return.txt', return_list)
        # np.savetxt(save_dir + '/training_data/model_return.txt', return_model_list)
        # np.savetxt(save_dir + '/training_data/training_loss.txt', training_loss)