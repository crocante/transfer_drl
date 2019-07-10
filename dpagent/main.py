import sys
sys.path.append('/data/GuoJiajia/PycharmProjects/transfer_drl/')
import os
import yaml
import gym
import tensorflow as tf
import time
import dpagent.ppo.core as core
from tt_utils.mpi_tools import mpi_fork, num_procs
from tt_utils.logx import EpochLogger
from dpagent.ppo.ppo import ppo
from dpagent.dp_new import DP_New

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file', type=str, default='halfcheetah')
    parser.add_argument('--env', type=str, default='HalfCheetah-v1')
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
    #ppo
    epochs = args.epochs
    steps_per_epoch = args.steps
    gamma = args.gamma
    seed = args.seed
    max_ep_len = 1000
    save_freq = 10
    train_pi_iters = 80
    train_v_iters = 80
    target_kl = 0.01
#     dyn_model
    num_fc_layers = params['dyn_model']['num_fc_layers']
    depth_fc_layers = params['dyn_model']['depth_fc_layers']
    batchsize = params['dyn_model']['batchsize']
    lr = params['dyn_model']['lr']
    nEpoch = params['dyn_model']['nEpoch']

#     mpi_fork(args.cpu)  # run parallel code with mpi

    # from utils.run_utils import setup_logger_kwargs
    # logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    # logger = EpochLogger(**logger_kwargs)
    # logger.save_config(locals())

    inputSize = 41+8
    outputSize = 41
    env = gym.make(args.env)
# #     env = normalize(env)  # apply normalize wrapper to env
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    agent = DP_New(inputSize, outputSize, batchsize, lr, num_fc_layers, nEpoch, depth_fc_layers, env, local_steps_per_epoch,
                   actor_critic=core.mlp_actor_critic, ac_kwargs=dict(hidden_sizes=[args.hid]*args.l))

    gpu_device = 0
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
        sess.graph.finalize()
        start_time = time.time()
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        for epoch in range(epochs):
            for t in range(local_steps_per_epoch):
                a, v_t, logp_t = sess.run(agent.get_action_ops, feed_dict={agent.x_ph: o.reshape(1, -1)})

                # save and log
                agent.buf.store(o, a, r, v_t, logp_t)
                # logger.store(VVals=v_t)

                o, r, d, _ = env.step(a[0])
                ep_ret += r
                ep_len += 1

                terminal = d or (ep_len == max_ep_len)
                if terminal or (t == local_steps_per_epoch - 1):
                    if not (terminal):
                        print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    last_val = r if d else sess.run(agent.v, feed_dict={agent.x_ph: o.reshape(1, -1)})
                    agent.buf.finish_path(last_val)
                    # if terminal:
                    #     # only save EpRet / EpLen if trajectory finished
                    #     logger.store(EpRet=ep_ret, EpLen=ep_len)
                    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

            # Save model
            # if (epoch % save_freq == 0) or (epoch == epochs - 1):
            #     logger.save_state({'env': env}, None)

            # Perform PPO update!
            agent.update(train_pi_iters, train_v_iters, target_kl)   #, logger
             # Log info about epoch
    #             logger.log_tabular('Epoch', epoch)
    #             logger.log_tabular('EpRet', with_min_and_max=True)
    #             logger.log_tabular('EpLen', average_only=True)
    #             logger.log_tabular('VVals', with_min_and_max=True)
    #             logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
    #             logger.log_tabular('LossPi', average_only=True)
    #             logger.log_tabular('LossV', average_only=True)
    #             logger.log_tabular('DeltaLossPi', average_only=True)
    #             logger.log_tabular('DeltaLossV', average_only=True)
    #             logger.log_tabular('Entropy', average_only=True)
    #             logger.log_tabular('KL', average_only=True)
    #             logger.log_tabular('ClipFrac', average_only=True)
    #             logger.log_tabular('StopIter', average_only=True)
    #             logger.log_tabular('Time', time.time() - start_time)
    #             logger.dump_tabular()
