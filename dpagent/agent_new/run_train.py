import sys
sys.path.append('/data/GuoJiajia/PycharmProjects/transfer_drl/')
import os
import yaml
import gym
import tensorflow as tf
import numpy as np
import time
from dpagent.dyn_model.data_manipulation import generate_training_data_inputs
from dpagent.dyn_model.data_manipulation import generate_training_data_outputs
from envs.normalized_env import normalize
from dpagent.agent_new.dp_new import DP_New
import dpagent.trpo.core as core
from tt_utils.logx import EpochLogger
from dpagent.dyn_model.policy_random import Policy_Random
from dpagent.dyn_model.collect_samples import CollectSamples
from dpagent.agent_adapt.sampler import collect_samples_new
# from tt_utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from tt_utils.mpi_tools import proc_id  #mpi_fork, mpi_avg, , mpi_statistics_scalar, num_procs
# from envs.half_cheetah_rand_direc import HalfCheetahRandDirecEnv
# from envs.half_cheetah_rand_vel import HalfCheetahRandVelEnv

def get_mean_std(dataX, dataY, dataZ):   #, agent
    mean_x = np.mean(dataX, axis=0)
    dataX = dataX - mean_x
    std_x = np.std(dataX, axis=0)
    dataX = np.nan_to_num(dataX / std_x)
    # agent.mean_x = mean_x
    # agent.std_x = std_x

    mean_y = np.mean(dataY, axis=0)
    dataY = dataY - mean_y
    std_y = np.std(dataY, axis=0)
    dataY = np.nan_to_num(dataY / std_y)
    # agent.mean_y = mean_y
    # agent.std_y = std_y

    mean_z = np.mean(dataZ, axis=0)
    dataZ = dataZ - mean_z
    std_z = np.std(dataZ, axis=0)
    dataZ = np.nan_to_num(dataZ / std_z)
    # agent.mean_z = mean_z
    # agent.std_z = std_z

    return dataX, mean_x, std_x, dataY, mean_y, std_y, dataZ, mean_z, std_z

def perform_rollouts(policy, num_rollouts, steps_per_rollout, CollectSamples, env):
    #collect training data by performing rollouts
    print("Beginning to do ", num_rollouts, " rollouts.")
    c = CollectSamples(env, policy)
    states, controls, rewards_list, rolloutrewards_list = c.collect_samples(num_rollouts, steps_per_rollout)

    print("Performed ", len(states), " rollouts, each with ", states[0].shape[0], " steps.")
    return states, controls, rewards_list, rolloutrewards_list

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file', type=str, default='halfcheetah')
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    yaml_path = os.path.abspath('../../yaml/' + args.yaml_file + '.yaml')
    assert (os.path.exists(yaml_path))
    with open(yaml_path, 'r') as f:
        params = yaml.load(f)
    # ppo
    steps_per_epoch = params['policy']['steps_per_epoch']
    gamma = params['policy']['gamma']
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
    # data_collection
    num_rollouts_train = params['data_collection']['num_rollouts_train']
    # steps
    steps_per_rollout_train = params['steps']['steps_per_rollout_train']
    # aggregation
    num_aggregation_iters = params['aggregation']['num_aggregation_iters']

    from tt_utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # mpi_fork(args.cpu)  # run parallel code with mpi

    save_dir = '../data/run_0728/halfcheetah'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir + '/losses')
        os.makedirs(save_dir + '/models')
        os.makedirs(save_dir + '/training_data')

    env = gym.make(args.env)
    # env = HalfCheetahRandDirecEnv()
    env = normalize(env)
    random_policy = Policy_Random(env)

    # local_steps_per_epoch = int(steps_per_epoch / num_procs())
    logger = EpochLogger(**logger_kwargs)
    # logger.save_config(locals())
    seed = args.seed
    seed += 10000 * proc_id()
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # gpu_device = 3
    # gpu_frac = 0.3
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
        dataX, mean_x, std_x, dataY, mean_y, std_y, dataZ, mean_z, std_z = \
            get_mean_std(dataX, dataY, dataZ)
        inputs = np.concatenate((dataX, dataY), axis=1)
        outputs = np.copy(dataZ)
        assert inputs.shape[0] == outputs.shape[0]
        inputSize = inputs.shape[1]
        outputSize = outputs.shape[1]

        dataX_new = np.zeros((0, dataX.shape[1]))
        dataY_new = np.zeros((0, dataY.shape[1]))
        dataZ_new = np.zeros((0, dataZ.shape[1]))

        agent = DP_New(inputSize, outputSize, lr, batchsize, num_fc_layers, depth_fc_layers, env,
                       steps_per_epoch, actor_critic=core.mlp_actor_critic,
                       ac_kwargs=dict(hidden_sizes=[args.hid] * args.l))
        agent.initialize(sess)
        # sess.graph.finalize()
        # pi_loss_list=[]
        # v_loss_list=[]
        # dyn_loss_list=[]
        ep_return = []
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        counter_agg_iters = 1
        start_time = time.time()

        while (counter_agg_iters < num_aggregation_iters+1):
            # how many full iterations of training-->rollouts-->aggregatedata to conduct
            start_time_iter = time.time()
            if (counter_agg_iters % 100 == 0):
                dataX_new_preprocessed = np.nan_to_num((dataX_new - mean_x) / std_x)
                dataY_new_preprocessed = np.nan_to_num((dataY_new - mean_y) / std_y)
                dataZ_new_preprocessed = np.nan_to_num((dataZ_new - mean_z) / std_z)

                inputs_new = np.concatenate((dataX_new_preprocessed, dataY_new_preprocessed), axis=1)
                # reward = reward.reshape(-1, 1)
                outputs_new = np.copy(dataZ_new_preprocessed)

                dyn_loss_avg = agent.update_dyn(inputs, outputs, inputs_new, outputs_new, nEpoch, fraction_use_new,
                                                fraction_use_val, target_loss, logger)
                # dyn_loss_list.append(dyn_loss_avg)
                agent.update_trpo(logger)
            else:
                # trpo
                for t in range(steps_per_epoch):
                    agent_outs = sess.run(agent.get_action_ops, feed_dict={agent.x_ph: o.reshape(1, -1)})
                    a, v_t, logp_t, info_t = agent_outs[0][0], agent_outs[1], agent_outs[2], agent_outs[3:]

                    # save and log
                    agent.buf.store(o, a, r, v_t, logp_t, info_t)
                    logger.store(VVals=v_t)

                    o, r, d, _ = env.step(a)
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
                            logger.store(EpRet=ep_ret, EpLen=ep_len)
                        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
                agent.update_trpo(logger)
            #ppo
            # for t in range(steps_per_epoch):
            #     a, v_t, logp_t = sess.run(agent.get_action_ops, feed_dict={agent.x_ph: o.reshape(1, -1)})
            #
            #     # save and log
            #     agent.buf.store(o, a, r, v_t, logp_t)
            #     logger.store(VVals=v_t)
            #
            #     next_o, r, d, _ = env.step(a[0])
            #     ep_ret += r
            #     ep_len += 1
            #     o = next_o
            #
            #     terminal = d or (ep_len == max_ep_len)
            #     if terminal or (t == steps_per_epoch - 1):
            #         if not (terminal):
            #             print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)
            #         # if trajectory didn't reach terminal state, bootstrap value target
            #         last_val = r if d else sess.run(agent.v, feed_dict={agent.x_ph: o.reshape(1, -1)})
            #         agent.buf.finish_path(last_val)
            #         if terminal:
            #             # only save EpRet / EpLen if trajectory finished
            #             logger.store(EpRet=ep_ret, EpLen=ep_len)
            #         ep_return.append(ep_ret)
            #         o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

            # Save model
            # if (epoch % save_freq == 0) or (epoch == epochs - 1):
            #     logger.save_state({'env': env}, None)

            # Log info about epoch
            logger.log_tabular('Epoch', counter_agg_iters)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('TotalEnvInteracts', (counter_agg_iters + 1) * steps_per_epoch)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('DeltaLossPi', average_only=True)
            logger.log_tabular('DeltaLossV', average_only=True)
            logger.log_tabular('KL', average_only=True)
            if agent.algo == 'trpo':
                logger.log_tabular('BacktrackIters', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()

            if ((counter_agg_iters+1) % 100==0):
                states, controls, rewards, rollout_rewards = collect_samples_new(num_rollouts_train, env, steps_per_rollout_train,
                                                                                 sess, logger, agent, max_ep_len)

                newdataX, newdataY = generate_training_data_inputs(states, controls)
                newdataZ = generate_training_data_outputs(states)

                dataX_new = np.concatenate((dataX_new, newdataX))
                dataY_new = np.concatenate((dataY_new, newdataY))
                dataZ_new = np.concatenate((dataZ_new, newdataZ))

            #save model
            # saver = tf.train.Saver(var_list=agent.dyn_params)
            # saver.save(sess, save_dir+"/models/dyn_model")
            # saver = tf.train.Saver(var_list=agent.policy_params)
            # saver.save(sess, save_dir + "/models/policy_model")
            if (counter_agg_iters % 200 == 0):
                agent.save(save_dir+"/models/"+str(counter_agg_iters))

            counter_agg_iters += 1

        print("total time:"+str(time.time()-start_time))

        np.save(save_dir + '/training_data/return.npy', ep_return)
        np.savetxt(save_dir + '/training_data/return.txt', ep_return)