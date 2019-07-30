import os
import tensorflow as tf
import yaml
import time
import numpy as np
from collections import OrderedDict
from envs.half_cheetah_rand_direc import HalfCheetahRandDirecEnv
from dpagent.dyn_model.data_manipulation import generate_training_data_inputs
from dpagent.dyn_model.data_manipulation import generate_training_data_outputs
from envs.normalized_env import normalize
from tt_utils.logx import EpochLogger
from tt_utils.mpi_tools import num_procs
from dpagent.agent_adapt.dp_adapt import DP_Adapt
import dpagent.agent_adapt.core as core
from dpagent.dyn_model.policy_random import Policy_Random
from dpagent.dyn_model.collect_samples import CollectSamples
from tkinter import _flatten
import matplotlib.pyplot as plt

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
    parser.add_argument('--env', type=str, default='HalfCheetah-v1')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.95)
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
    num_rollouts_train = params['data_collection']['num_rollouts_train']  # 33
    # steps
    steps_per_rollout_train = params['steps']['steps_per_rollout_train']
    steps_rollout_train = params['steps']['steps_rollout_train']
    noiseToSignal = 0.01
    from tt_utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    save_dir = 'data/run_0702/restore_dyn'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir + '/losses')
        os.makedirs(save_dir + '/models')
        os.makedirs(save_dir + '/training_data')

    env = HalfCheetahRandDirecEnv(1)
    env = normalize(env)    # apply normalize wrapper to env
    random_policy = Policy_Random(env)

    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    # model_logger = EpochLogger(**logger_kwargs)
    logger = EpochLogger(**logger_kwargs)

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
        states, controls, rewards, rollout_rewards = perform_rollouts(random_policy, num_rollouts_train * 2,
                                                                      steps_per_rollout_train,
                                                                      CollectSamples, env)
        dataX, dataY = generate_training_data_inputs(states, controls)
        dataZ = generate_training_data_outputs(states)
        dataX, mean_x, std_x, dataY, mean_y, std_y, dataZ, mean_z, std_z = \
            get_mean_std(dataX, dataY, dataZ)
        inputs = np.concatenate((dataX, dataY), axis=1)
        # reward = reward.reshape(-1, 1)
        outputs = np.copy(dataZ)
        assert inputs.shape[0] == outputs.shape[0]
        inputSize = inputs.shape[1]
        outputSize = outputs.shape[1]
        #
        # dataX_new = np.zeros((0, dataX.shape[1]))
        # dataY_new = np.zeros((0, dataY.shape[1]))
        # dataZ_new = np.zeros((0, dataZ.shape[1]))

        saver = tf.train.import_meta_graph('data/run_0726/halfcheetah_kl_0.05/origin/models/800.meta')
        saver.restore(sess, tf.train.latest_checkpoint('data/run_0726/halfcheetah_kl_0.05/origin/models'))

        # graph_ = tf.get_default_graph().get_all_collection_keys()
        params = tf.get_default_graph().get_collection("trainable_variables")

        odict = OrderedDict()
        for var in params:
            odict[var.name] = var
        agent = DP_Adapt(odict, inputSize, outputSize, env, steps_per_epoch, lr, batchsize, actor_critic=core.mlp_actor_critic,
                         mean_x=mean_x, std_x=std_x, mean_y=mean_y, std_y=std_y, mean_z=mean_z, std_z=std_z)
        agent.initialize(sess)
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        x = []
        z = []
        x.append(o[0])
        z.append(o[1])
        for t in range(local_steps_per_epoch):
            a, v_t, logp_t = sess.run(agent.get_action_ops, feed_dict={agent.x_ph: o.reshape(1, -1)})

            agent.buf.store(o, a, r, v_t, logp_t)
            logger.store(VVals=v_t)
            observation = np.copy(o)
            action = np.copy(a[0])
            # observations.append(observation)
            # actions.append(action)

            next_o, r, d, _ = env.step(a[0])
            x.append(next_o[0])
            z.append(next_o[1])
            next_o_model = agent.do_forward(observation, action)
            rewards.append(r)
            ep_ret += r
            ep_len += 1
            o = next_o

            terminal = d or (ep_len == max_ep_len)
            if terminal or (t == local_steps_per_epoch - 1):
                if not (terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = r if d else sess.run(agent.v, feed_dict={agent.x_ph: o.reshape(1, -1)})
                agent.buf.finish_path(last_val)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                    # ep_return.append(ep_ret)
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
                break
        num = np.arange(len(x))
        plt.plot(num, x)
        plt.legend()
        plt.show()
        # avg_val_loss, validation_loss_list, pred_x_list, pred_z_list = agent.run_validation(inputs, outputs)
        # x_list = dataZ[:, 0]
        # z_list = dataZ[:, 0]
