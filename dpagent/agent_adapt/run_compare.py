import os
import tensorflow as tf
import yaml
import time
import numpy as np
from collections import OrderedDict
# from envs.half_cheetah_rand_direc import HalfCheetahRandDirecEnv
from envs.half_cheetah_rand_vel import HalfCheetahRandVelEnv
from dpagent.dyn_model.data_manipulation import generate_training_data_inputs
from dpagent.dyn_model.data_manipulation import generate_training_data_outputs
from envs.normalized_env import normalize
from tt_utils.logx import EpochLogger
from tt_utils.mpi_tools import num_procs
from dpagent.agent_adapt.dp_adapt import DP_Adapt
import dpagent.agent_adapt.core as core
from dpagent.agent_adapt.sampler import get_model_samples, get_samples, get_samples_with_buf

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

    yaml_path = os.path.abspath('../../yaml/' + args.yaml_file + '.yaml')
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
    # aggregation
    num_aggregation_iters = params['aggregation']['num_aggregation_iters']
    inner_aggregation_iters_k = params['aggregation']['inner_aggregation_iters_k']

    from tt_utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    save_dir = '../data/run_0624/compare'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir + '/losses')
        os.makedirs(save_dir + '/models')
        os.makedirs(save_dir + '/training_data')

    env = HalfCheetahRandVelEnv(-3)
    env = normalize(env)    # apply normalize wrapper to env

    inputSize = env.observation_space.shape[0] + env.action_space.shape[0]
    outputSize = env.observation_space.shape[0] + 1
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    model_logger = EpochLogger(**logger_kwargs)
    logger = EpochLogger(**logger_kwargs)

    gpu_device = 2
    gpu_frac = 0.5
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
    config = tf.ConfigProto(gpu_options=gpu_options,
                            log_device_placement=False,
                            allow_soft_placement=True,
                            inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1)

    with tf.Session(config=config) as sess:
        # agent.initialize(sess)
        # saver = tf.train.Saver()
        # saver.restore(sess, "data/run_0613/models/500")
        saver = tf.train.import_meta_graph('../data/run_0624/env_vel_3/models/1000.meta')
        saver.restore(sess, tf.train.latest_checkpoint('../data/run_0624/env_vel_3/models'))

        # graph_ = tf.get_default_graph().get_all_collection_keys()
        params = tf.get_default_graph().get_collection("trainable_variables")

        odict = OrderedDict()
        for var in params:
            odict[var.name] = var
        agent = DP_Adapt(odict, inputSize, outputSize, env, local_steps_per_epoch, lr, actor_critic=core.mlp_actor_critic)

        # graph_ = tf.get_default_graph().get_all_collection_keys()
        # params_ = tf.get_default_graph().get_collection("train_op")
        # params_1 = tf.get_default_graph().get_collection("trainable_variables")
        # params_2 = tf.get_default_graph().get_collection("variables")
        # params_3 = tf.all_variables()

        agent.initialize(sess)
        pi_loss_list=[]
        v_loss_list=[]
        dyn_loss_list=[]
        ep_return = []
        counter_agg_iters = 0
        # start_time = time.time()
        while (counter_agg_iters < num_aggregation_iters+1):
            start_time_iter = time.time()
            #interact with real world
            observations, actions, rewards, ep_ret = get_samples_with_buf(env, local_steps_per_epoch, sess, agent, logger, max_ep_len)
            ep_return.append(ep_ret)
            observations = np.array(observations)
            actions = np.array(actions)

            dataX, dataY = generate_training_data_inputs(observations, actions)
            dataZ, reward = generate_training_data_outputs(observations, rewards)
            dataX, mean_x, std_x, dataY, mean_y, std_y, dataZ, mean_z, std_z, reward, mean_r, std_r = \
                get_mean_std(dataX, dataY, dataZ, reward, agent)

            inputs = np.concatenate((dataX, dataY), axis=1)
            reward = reward.reshape(-1, 1)
            outputs = np.concatenate((dataZ, reward), axis=1)
            assert inputs.shape[0] == outputs.shape[0]

            pi_loss, v_loss = agent.update_outer_policy(train_pi_iters, train_v_iters, target_kl, logger)  # , logger
            dyn_avg_loss = agent.update_dyn(inputs, outputs, nEpoch, batchsize, logger)
            pi_loss_list.append(pi_loss)
            v_loss_list.append(v_loss)
            dyn_loss_list.append(dyn_avg_loss)
            counter_agg_iters += 1
            if (counter_agg_iters % 100 == 0):
                agent.save(save_dir+"/models/"+str(counter_agg_iters))
            logger.log_tabular('Epoch', counter_agg_iters)
            # logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpRet', average_only=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('VVals', average_only=True)
            # logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('TotalEnvInteracts', (counter_agg_iters + 1) * steps_per_epoch)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('LossDyn', average_only=True)
            logger.log_tabular('DeltaLossPi', average_only=True)
            logger.log_tabular('DeltaLossV', average_only=True)
            # logger.log_tabular('Entropy', average_only=True)
            # logger.log_tabular('KL', average_only=True)
            # logger.log_tabular('ClipFrac', average_only=True)
            logger.log_tabular('StopIter', average_only=True)
            logger.log_tabular('Time', time.time() - start_time_iter)
            logger.dump_tabular()

        np.save(save_dir + '/training_data/return.npy', ep_return)
        np.savetxt(save_dir + '/training_data/return.txt', ep_return)
        np.save(save_dir + '/losses/LossPi.npy', pi_loss_list)
        np.savetxt(save_dir + '/losses/LossPi.txt', pi_loss_list)
        np.save(save_dir + '/losses/LossV.npy', v_loss_list)
        np.savetxt(save_dir + '/losses/LossV.txt', v_loss_list)
        np.save(save_dir + '/losses/dyn_losses.npy', dyn_loss_list)
        np.savetxt(save_dir + '/losses/dyn_losses.txt', dyn_loss_list)