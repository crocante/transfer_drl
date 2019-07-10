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
from envs.half_cheetah_rand_direc import HalfCheetahRandDirecEnv
# from envs.half_cheetah_rand_vel import HalfCheetahRandVelEnv
from envs.normalized_env import normalize
from dpagent.agent_new.dp_new import DP_New
import dpagent.agent_new.core as core
from tt_utils.logx import EpochLogger
# from tt_utils.mpi_tf import MpiAdamOptimizer, sync_all_params
# from tt_utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

def normalization(data, mean, std):
    data = data - mean
    data = np.nan_to_num(data / std)
    return data

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file', type=str, default='halfcheetah')
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    yaml_path = os.path.abspath('../../yaml/' + args.yaml_file + '.yaml')
    assert (os.path.exists(yaml_path))
    with open(yaml_path, 'r') as f:
        params = yaml.load(f)
    #ppo
    steps_per_epoch = params['policy']['steps_per_epoch']
    gamma = params['policy']['gamma']
    max_ep_len = params['policy']['max_ep_len']
    train_pi_iters = params['policy']['train_pi_iters']
    train_v_iters = params['policy']['train_v_iters']
    target_kl = params['policy']['target_kl']
    #dyn_model
    num_fc_layers = params['dyn_model']['num_fc_layers']
    depth_fc_layers = params['dyn_model']['depth_fc_layers']
    batchsize = params['dyn_model']['batchsize']
    lr = params['dyn_model']['lr']
    nEpoch = params['dyn_model']['nEpoch']
    # aggregation
    num_aggregation_iters = params['aggregation']['num_aggregation_iters']

    # mpi_fork(args.cpu)  # run parallel code with mpi

    from tt_utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    save_dir = '../data/run_0702/change_env'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir + '/losses')
        os.makedirs(save_dir + '/models')
        os.makedirs(save_dir + '/training_data')
    # env = gym.make(args.env)
    env = HalfCheetahRandDirecEnv()
    env = normalize(env)
    inputSize = env.observation_space.shape[0] + env.action_space.shape[0]
    outputSize = env.observation_space.shape[0] + 1
    # local_steps_per_epoch = int(steps_per_epoch / num_procs())
    logger = EpochLogger(**logger_kwargs)
    # logger.save_config(locals())

    agent = DP_New(inputSize, outputSize, lr, batchsize, num_fc_layers, depth_fc_layers, env,
                   steps_per_epoch, actor_critic=core.mlp_actor_critic,
                   ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), seed=args.seed)

    gpu_device = 7
    gpu_frac = 0.3
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
    config = tf.ConfigProto(gpu_options=gpu_options,
                            log_device_placement=False,
                            allow_soft_placement=True,
                            inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1)

    with tf.Session(config=config) as sess:
        agent.initialize(sess)
        # sess.graph.finalize()
        pi_loss_list=[]
        v_loss_list=[]
        dyn_loss_list=[]
        ep_return = []
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        counter_agg_iters = 0
        start_time = time.time()

        while (counter_agg_iters < num_aggregation_iters+1):
            observations = []
            actions = []
            rewards = []

            start_time_iter = time.time()
            for t in range(steps_per_epoch):
                a, v_t, logp_t = sess.run(agent.get_action_ops, feed_dict={agent.x_ph: o.reshape(1, -1)})

                # save and log
                agent.buf.store(o, a, r, v_t, logp_t)
                logger.store(VVals=v_t)
                observation = np.copy(o)
                action = np.copy(a[0])
                observations.append(observation)
                actions.append(action)

                next_o, r, d, _ = env.step(a[0])
                rewards.append(r)
                #store o,a,next_o,r

                ep_ret += r
                ep_len += 1
                o = next_o

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
                    ep_return.append(ep_ret)
                    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

            # Save model
            # if (epoch % save_freq == 0) or (epoch == epochs - 1):
            #     logger.save_state({'env': env}, None)

            observations = np.array(observations)
            actions = np.array(actions)

            dataX, dataY = generate_training_data_inputs(observations, actions)
            dataZ, reward = generate_training_data_outputs(observations, rewards)

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

            mean_r = np.mean(reward, axis=0)
            reward = reward - mean_r
            std_r = np.std(reward, axis=0)
            reward = np.nan_to_num(reward / std_r)

            inputs = np.concatenate((dataX, dataY), axis=1)
            reward = reward.reshape(-1, 1)
            outputs = np.concatenate((dataZ, reward), axis=1)
            assert inputs.shape[0] == outputs.shape[0]
            # Perform update!
            dyn_losses, dyn_loss_avg, pi_loss, v_loss = agent.update(inputs, outputs, mean_x, std_x, mean_y, std_y,
                                                                     mean_z, std_z, mean_r, std_r, nEpoch, train_pi_iters,
                                                                     train_v_iters, target_kl, logger)  #, logger
            dyn_loss_list.append(dyn_losses)
            pi_loss_list.append(pi_loss)
            v_loss_list.append(v_loss)
            counter_agg_iters+=1

            #save model
            # saver = tf.train.Saver(var_list=agent.dyn_params)
            # saver.save(sess, save_dir+"/models/dyn_model")
            # saver = tf.train.Saver(var_list=agent.policy_params)
            # saver.save(sess, save_dir + "/models/policy_model")
            if (counter_agg_iters % 200 == 0):
                agent.save(save_dir+"/models/"+str(counter_agg_iters))
            # Log info about epoch
            logger.log_tabular('Epoch', counter_agg_iters)
            # logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpRet', average_only=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('VVals', average_only=True)
            # logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('TotalEnvInteracts', (counter_agg_iters + 1) * steps_per_epoch)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('LossDyn', dyn_loss_avg)
            logger.log_tabular('DeltaLossPi', average_only=True)
            logger.log_tabular('DeltaLossV', average_only=True)
            # logger.log_tabular('Entropy', average_only=True)
            # logger.log_tabular('KL', average_only=True)
            # logger.log_tabular('ClipFrac', average_only=True)
            logger.log_tabular('StopIter', average_only=True)
            logger.log_tabular('Time', time.time() - start_time_iter)
            logger.dump_tabular()

        print("total time:"+str(time.time()-start_time))

        np.save(save_dir + '/training_data/return.npy', ep_return)
        np.savetxt(save_dir + '/training_data/return.txt', ep_return)
        np.save(save_dir + '/losses/LossPi.npy', pi_loss_list)
        np.savetxt(save_dir + '/losses/LossPi.txt', pi_loss_list)
        np.save(save_dir + '/losses/LossV.npy', v_loss_list)
        np.savetxt(save_dir + '/losses/LossV.txt', v_loss_list)
        np.save(save_dir + '/losses/dyn_losses.npy', dyn_loss_list)
        np.savetxt(save_dir + '/losses/dyn_losses.txt', dyn_loss_list)
