import os
import tensorflow as tf
import yaml
import time
from tkinter import _flatten
import numpy as np
import gym
from collections import OrderedDict
from envs.half_cheetah_rand_direc import HalfCheetahRandDirecEnv
from envs.normalized_env import normalize
from dpagent.dyn_model.data_manipulation import generate_training_data_inputs
from dpagent.dyn_model.data_manipulation import generate_training_data_outputs
from tt_utils.logx import EpochLogger
# from tt_utils.mpi_tools import num_procs, mpi_fork, proc_id
# from tt_utils.mpi_tf import sync_all_params
from dpagent.agent_adapt.dp_adapt import DP_Adapt
import dpagent.agent_adapt.core as core
from dpagent.agent_adapt.sampler import get_samples, get_reward
from dpagent.dyn_model.collect_samples import CollectSamples
from dpagent.dyn_model.policy_random import Policy_Random

def perform_rollouts(policy, num_rollouts, steps_per_rollout, CollectSamples, env):
    #collect training data by performing rollouts
    print("Beginning to do ", num_rollouts, " rollouts.")
    c = CollectSamples(env, policy)
    states, controls, rewards_list, rolloutrewards_list = c.collect_samples(num_rollouts, steps_per_rollout)

    print("Performed ", len(states), " rollouts, each with ", states[0].shape[0], " steps.")
    return states, controls, rewards_list, rolloutrewards_list

def get_mean_std(dataX, dataY, dataZ):
    mean_x = np.mean(dataX, axis=0)
    dataX = dataX - mean_x
    std_x = np.std(dataX, axis=0)
    dataX = np.nan_to_num(dataX / std_x)

    mean_y = np.mean(dataY, axis=0)
    dataY = dataY - mean_y
    std_y = np.std(dataY, axis=0)
    dataY = np.nan_to_num(dataY / std_y)

    mean_z = np.mean(dataZ, axis=0)
    dataZ = dataZ - mean_z
    std_z = np.std(dataZ, axis=0)
    dataZ = np.nan_to_num(dataZ / std_z)

    return dataX, mean_x, std_x, dataY, mean_y, std_y, dataZ, mean_z, std_z

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file', type=str, default='halfcheetah_adapt')
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    yaml_path = os.path.abspath('../../yaml/' + args.yaml_file + '.yaml')
    assert (os.path.exists(yaml_path))
    with open(yaml_path, 'r') as f:
        params = yaml.load(f)
    # ppo
    steps_per_epoch = params['policy']['steps_per_epoch']
    seed = args.seed
    max_ep_len = params['policy']['max_ep_len']
    train_pi_iters = params['policy']['train_pi_iters']
    train_v_iters = params['policy']['train_v_iters']
    target_kl = params['policy']['target_kl']

    # dyn_model
    num_fc_layers = params['dyn_model']['num_fc_layers']
    depth_fc_layers = params['dyn_model']['depth_fc_layers']
    batchsize = params['dyn_model']['batchsize']
    lr = params['dyn_model']['lr']
    nEpoch = params['dyn_model']['nEpoch']
    fraction_use_new = params['dyn_model']['fraction_use_new']
    fraction_use_val = params['dyn_model']['fraction_use_val']
    target_loss = params['dyn_model']['target_loss']
    # aggregation
    num_aggregation_iters = params['aggregation']['num_aggregation_iters']
    inner_aggregation_iters_k = params['aggregation']['inner_aggregation_iters_k']
    # noise
    make_aggregated_dataset_noisy = params['noise']['make_aggregated_dataset_noisy']
    # data collection
    num_rollouts_train = params['data_collection']['num_rollouts_train']  #33
    # steps
    steps_per_rollout_train = params['steps']['steps_per_rollout_train']
    steps_rollout_train = params['steps']['steps_rollout_train']

    noiseToSignal = 0.01
    from tt_utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # mpi_fork(args.cpu)  # run parallel code with mpi
    # seed += 10000 * proc_id()
    # tf.set_random_seed(seed)
    # np.random.seed(seed)

    save_dir = '../data/run_0727/halfcheetah/adapt/0721'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir + '/losses')
        os.makedirs(save_dir + '/models')
        os.makedirs(save_dir + '/training_data')

    env = gym.make(args.env)
    # env = HalfCheetahRandDirecEnv()
    env = normalize(env)    # apply normalize wrapper to env
    random_policy = Policy_Random(env)

    # inputSize = env.observation_space.shape[0] + env.action_space.shape[0]
    # outputSize = env.observation_space.shape[0] + 1
    # local_steps_per_epoch = int(steps_per_epoch / num_procs())
    model_logger = EpochLogger(**logger_kwargs)
    logger = EpochLogger(**logger_kwargs)

    # gpu_device = 1
    # gpu_frac = 0.4
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
    # config = tf.ConfigProto(gpu_options=gpu_options,
    #                         log_device_placement=False,
    #                         allow_soft_placement=True,
    #                         inter_op_parallelism_threads=1,
    #                         intra_op_parallelism_threads=1)

    # with tf.Session(config=config) as sess:
    with tf.Session() as sess:
        states, controls, rewards, rollout_rewards = perform_rollouts(random_policy, num_rollouts_train * 10,
                                                                      steps_per_rollout_train,
                                                                      CollectSamples, env)
        dataX, dataY = generate_training_data_inputs(states, controls)
        dataZ = generate_training_data_outputs(states)
        dataX_preprocessed, mean_x, std_x, dataY_preprocessed, mean_y, std_y, dataZ_preprocessed, mean_z, std_z = \
            get_mean_std(dataX, dataY, dataZ)
        inputs = np.concatenate((dataX_preprocessed, dataY_preprocessed), axis=1)
        # reward = reward.reshape(-1, 1)
        outputs = np.copy(dataZ_preprocessed)
        assert inputs.shape[0] == outputs.shape[0]
        inputSize = inputs.shape[1]
        outputSize = outputs.shape[1]

        dataX_new = np.zeros((0, dataX.shape[1]))
        dataY_new = np.zeros((0, dataY.shape[1]))
        dataZ_new = np.zeros((0, dataZ.shape[1]))

        saver = tf.train.import_meta_graph('../data/run_0728/halfcheetah/models/800.meta')
        saver.restore(sess, tf.train.latest_checkpoint('../data/run_0728/halfcheetah/models'))

        # graph_ = tf.get_default_graph().get_all_collection_keys()
        params = tf.get_default_graph().get_collection("trainable_variables")

        odict = OrderedDict()
        for var in params:
            odict[var.name] = var
        agent = DP_Adapt(odict, inputSize, outputSize, env, steps_per_epoch, lr, batchsize, actor_critic=core.mlp_actor_critic,
                         mean_x=mean_x, std_x=std_x, mean_y=mean_y, std_y=std_y, mean_z=mean_z, std_z=std_z)
        agent.initialize(sess)

        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        counter_agg_iters = 0
        dyn_loss_list = []
        return_list = []

        while (counter_agg_iters < num_aggregation_iters+1):
            #################train model###############
            print("current iteration:", counter_agg_iters)
            dataX_new_preprocessed = np.nan_to_num((dataX_new - mean_x) / std_x)
            dataY_new_preprocessed = np.nan_to_num((dataY_new - mean_y) / std_y)
            dataZ_new_preprocessed = np.nan_to_num((dataZ_new - mean_z) / std_z)

            inputs_new = np.concatenate((dataX_new_preprocessed, dataY_new_preprocessed), axis=1)
            # reward = reward.reshape(-1, 1)
            outputs_new = np.copy(dataZ_new_preprocessed)

            dyn_loss_avg = agent.update_dyn(inputs, outputs, inputs_new, outputs_new, nEpoch,
                                            fraction_use_new, fraction_use_val, target_loss, logger)
            dyn_loss_list.append(dyn_loss_avg)
            ##############train policy#############
            # mpi_fork(args.cpu)  # run parallel code with mpi
            max_ep_len = 50
            for k in range(inner_aggregation_iters_k):
                if(dataX_new.shape[0]==0):
                    o, r, d, ep_ret, ep_len = dataX[np.random.choice(dataX.shape[0]),:], 0, False, 0, 0
                else:
                    o, r, d, ep_ret, ep_len = dataX_new[np.random.choice(dataX_new.shape[0]),:], 0, False, 0, 0
                for t in range(steps_per_epoch):
                    a, v_t, logp_t = sess.run(agent.get_action_ops, feed_dict={agent.x_ph: o.reshape(1, -1)})

                    # save and log
                    agent.buf.store(o, a, r, v_t, logp_t)
                    model_logger.store(VVals=v_t)

                    observation = np.copy(o)
                    action = np.copy(a[0])
                    o = agent.do_forward(observation, action)
                    next_observation = np.copy(o)
                    r = get_reward(observation, action, next_observation)
                    ep_ret += r
                    ep_len += 1

                    terminal = d or (ep_len == max_ep_len)
                    if terminal or (t == steps_per_epoch - 1):
                        if not (terminal):
                            print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)
                        # if trajectory didn't reach terminal state, bootstrap value target
                        last_val = r if d else sess.run(agent.v, feed_dict={agent.x_ph: o.reshape(1, -1)})
                        agent.buf.finish_path(last_val)
                        if terminal:
                            # only save EpRet / EpLen if trajectory finished
                            model_logger.store(AvgRew_model=ep_ret)   #, EpLen=ep_len
                            return_list.append(ep_ret)
                        if (dataX_new.shape[0] == 0):
                            o, r, d, ep_ret, ep_len = dataX[np.random.choice(dataX.shape[0]), :], 0, False, 0, 0
                            # print("\n=== Step {} ===".format(t), '\nreset model environment')
                        else:
                            o, r, d, ep_ret, ep_len = dataX_new[np.random.choice(dataX_new.shape[0]), :], 0, False, 0, 0
                            # print("\n=== Step {} ===".format(t), '\nreset model environment')

                agent.update_inner_policy(model_logger)
                # pi_loss_model_list.append(pi_loss)
                # v_loss_model_list.append(v_loss)
                model_logger.log_tabular('Epoch', k)
                model_logger.log_tabular('AvgRew_model', average_only=True)
                model_logger.log_tabular('VVals', average_only=True)
                model_logger.log_tabular('LossPi_model', average_only=True)
                model_logger.log_tabular('LossV_model', average_only=True)
                model_logger.log_tabular('Entropy_model', average_only=True)
                model_logger.log_tabular('KL_model', average_only=True)
                model_logger.log_tabular('ClipFrac_model', average_only=True)
                model_logger.log_tabular('DeltaLossPi_model', average_only=True)
                model_logger.log_tabular('DeltaLossV_model', average_only=True)
                model_logger.dump_tabular()
            ################train policy in real env############
            max_ep_len = 200
            states, controls, rewards, rollout_rewards = get_samples(env, steps_rollout_train, sess,
                                                                     agent, logger, max_ep_len)
            agent.update_outer_policy(logger)
            return_list.append(rollout_rewards)
            newdataX, newdataY = generate_training_data_inputs(states, controls)
            newdataZ = generate_training_data_outputs(states)

            dataX_new = np.concatenate((dataX_new, newdataX))
            dataY_new = np.concatenate((dataY_new, newdataY))
            dataZ_new = np.concatenate((dataZ_new, newdataZ))

            # if(counter_agg_iters%50==0):
            #     np.savetxt(save_dir + '/training_data/env_x_' + str(counter_agg_iters)+'.txt', list(_flatten(x_list)))
            #     np.savetxt(save_dir + '/training_data/env_z_' + str(counter_agg_iters)+'.txt', list(_flatten(z_list)))
            #     # np.savetxt(save_dir + '/training_data/model_x_' + str(counter_agg_iters)+'.txt', list(_flatten(x_model_list)))
            #     # np.savetxt(save_dir + '/training_data/model_z_' + str(counter_agg_iters)+'.txt', list(_flatten(z_model_list)))
            #     np.savetxt(save_dir + '/training_data/env_rewards_' + str(counter_agg_iters)+'.txt', list(_flatten(list_rewards)))
                # np.savetxt(save_dir + '/training_data/model_rewards_' + str(counter_agg_iters)+'.txt', list(_flatten(rewards_model_list)))

            counter_agg_iters = counter_agg_iters + 1
            if (counter_agg_iters % 20 == 0):
                agent.save(save_dir + "/models/" + str(counter_agg_iters))
            logger.log_tabular('Epoch', counter_agg_iters)
            # logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpRet', average_only=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('VVals', average_only=True)
            # logger.log_tabular('VVals', with_min_and_max=True)
            # logger.log_tabular('TotalEnvInteracts', (counter_agg_iters + 1) * steps_per_epoch)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('LossDyn', average_only=True)
            logger.log_tabular('DeltaLossPi', average_only=True)
            logger.log_tabular('DeltaLossV', average_only=True)
            logger.log_tabular('Entropy', average_only=True)
            logger.log_tabular('KL', average_only=True)
            logger.log_tabular('ClipFrac', average_only=True)
            logger.log_tabular('StopIter', average_only=True)
            logger.dump_tabular()
        np.savetxt(save_dir + '/training_data/env_return.txt', list(_flatten(return_list)))
        # np.savetxt(save_dir + '/training_data/model_return.txt', return_model_list)
        np.savetxt(save_dir + '/training_data/training_loss.txt', list(_flatten(dyn_loss_list)))