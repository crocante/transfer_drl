import numpy as np
import tensorflow as tf
import numpy.random as npr
import math
from collections import OrderedDict

import dpagent.agent_new.core as core
from dpagent.dyn_model.feedforward_network import feedforward_network
from dpagent.ppo.ppo import PPOBuffer
# from tt_utils.logx import EpochLogger
from tt_utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from tt_utils.mpi_tools import mpi_avg, proc_id        #, mpi_fork, mpi_statistics_scalar, num_procs

class DP_New():
    def __init__(self, inputSize, outputSize, learning_rate, batchsize, num_fc_layers, depth_fc_layers,
                 env, steps_per_epoch, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
                 gamma=0.99, clip_ratio=0.2, pi_lr=3e-4, vf_lr=1e-3, train_pi_iters=80, train_v_iters=80,
                 lam=0.97, target_kl=0.01, mean_x=0, mean_y=0, mean_z=0, mean_r=0,
                 std_x=0, std_y=0, std_z=0, std_r=0, tf_datatype=tf.float32):
        # with tf.variable_scope('dyn'):
        self.init_dyn(inputSize, outputSize, learning_rate, num_fc_layers, depth_fc_layers, tf_datatype)  #, scope='dyn'

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        # with tf.variable_scope('policy'):
        self.init_ppo(env, obs_dim, act_dim, actor_critic, ac_kwargs, seed, steps_per_epoch, gamma,
                          lam, clip_ratio, pi_lr, vf_lr)  #, scope='policy'
        #networks parameters
        # self.params = tf.trainable_variables()
        # dynamic model
        self.batchsize = batchsize
        self.mean_x = mean_x
        self.mean_y = mean_y
        self.mean_z = mean_z
        self.mean_r = mean_r
        self.std_x = std_x
        self.std_y = std_y
        self.std_z = std_z
        self.std_r = std_r
        #policy model
        # self.train_pi_iters = train_pi_iters
        # self.train_v_iters = train_v_iters
        # self.epochs = epochs
        self.lam = lam
        # self.max_ep_len = max_ep_len
        # self.target_kl = target_kl
        # self.save_freq = save_freq
        # self.logger = EpochLogger(**logger_kwargs)
        # self.logger.save_config(locals())

    def init_dyn(self, inputSize, outputSize, learning_rate, num_fc_layers, depth_fc_layers, tf_datatype):  #, scope
        print("initializing dynamics model......")
        self.x_ = tf.placeholder(tf_datatype, shape=[None, inputSize], name='x')  # inputs
        self.z_ = tf.placeholder(tf_datatype, shape=[None, outputSize], name='z')  # labels

        # with tf.variable_scope(scope):
        # forward pass
        with tf.variable_scope('dyn'):
            self.curr_nn_output = feedforward_network(self.x_, inputSize, outputSize, num_fc_layers,
                                                      depth_fc_layers, tf_datatype, scope='dyn' )

        # loss
        self.mse_ = tf.reduce_mean(tf.square(self.z_ - self.curr_nn_output))

        # Compute gradients and update parameters
        self.opt = tf.train.AdamOptimizer(learning_rate)
        self.theta = tf.trainable_variables()
        self.gv = [(g, v) for g, v in
                   self.opt.compute_gradients(self.mse_, self.theta)
                   if g is not None]
        self.train_step = self.opt.apply_gradients(self.gv)

    def init_ppo(self, env, obs_dim, act_dim, actor_critic, ac_kwargs, seed, local_steps_per_epoch, gamma,
                 lam, clip_ratio, pi_lr, vf_lr):
        print("initializing policy model........")
        seed += 10000 * proc_id()
        tf.set_random_seed(seed)
        np.random.seed(seed)
        # Share information about action space with policy architecture
        ac_kwargs['action_space'] = env.action_space
        # Inputs to computation graph
        self.x_ph, self.a_ph = core.placeholders_from_spaces(env.observation_space, env.action_space)
        self.adv_ph, self.ret_ph, self.logp_old_ph = core.placeholders(None, None, None)
        # Main outputs from computation graph
        # with tf.variable_scope(scope):
        self.pi, self.logp, self.logp_pi, self.v = actor_critic(self.x_ph, self.a_ph, **ac_kwargs)
        # Need all placeholders in *this* order later (to zip with data from buffer)
        self.all_phs = [self.x_ph, self.a_ph, self.adv_ph, self.ret_ph, self.logp_old_ph]

        # Every step, get: action, value, and logprob
        self.get_action_ops = [self.pi, self.v, self.logp_pi]

        # Experience buffer
        self.buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

        # Count variables
        var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])
        print('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

        # PPO objectives
        ratio = tf.exp(self.logp - self.logp_old_ph)  # pi(a|s) / pi_old(a|s)
        min_adv = tf.where(self.adv_ph > 0, (1 + clip_ratio) * self.adv_ph, (1 - clip_ratio) * self.adv_ph)
        self.pi_loss = -tf.reduce_mean(tf.minimum(ratio * self.adv_ph, min_adv))
        self.v_loss = tf.reduce_mean((self.ret_ph - self.v) ** 2)

        # Info (useful to watch during learning)
        self.approx_kl = tf.reduce_mean(self.logp_old_ph - self.logp)  # a sample estimate for KL-divergence, easy to compute
        self.approx_ent = tf.reduce_mean(-self.logp)  # a sample estimate for entropy, also easy to compute
        clipped = tf.logical_or(ratio > (1 + clip_ratio), ratio < (1 - clip_ratio))
        self.clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

        # Optimizers
        self.train_pi = MpiAdamOptimizer(learning_rate=pi_lr).minimize(self.pi_loss)
        self.train_v = MpiAdamOptimizer(learning_rate=vf_lr).minimize(self.v_loss)

    def initialize(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(sync_all_params())

    def update(self, dataX, dataZ, mean_x, std_x, mean_y, std_y, mean_z, std_z, mean_r, std_r,
               nEpoch, train_pi_iters, train_v_iters, target_kl, logger):   #, logger
        # pi_loss_list=[]
        # v_loss_list=[]
        self.update_mean_std(mean_x, mean_y, mean_z, mean_r, std_x, std_y, std_z, std_r)
        training_loss_list = []
        range_of_indeces = np.arange(dataX.shape[0])
        nData = dataX.shape[0]
        batchsize = int(self.batchsize)
        # Training
        for i in range(nEpoch):
            avg_loss = 0
            num_batches = 0
            indeces = npr.choice(range_of_indeces, size=(dataX.shape[0],), replace=False)
            for batch in range(int(math.floor(nData / batchsize))):
                # walk through the randomly reordered "old data"
                dataX_batch = dataX[indeces[batch * batchsize:(batch + 1) * batchsize], :]
                dataZ_batch = dataZ[indeces[batch * batchsize:(batch + 1) * batchsize], :]

                # one iteration of feedforward training
                _, loss, output, true_output = self.sess.run([self.train_step, self.mse_, self.curr_nn_output, self.z_],
                                                             feed_dict={self.x_: dataX_batch, self.z_: dataZ_batch})
                training_loss_list.append(loss)
                avg_loss += loss
                num_batches += 1

            # if ((i % 10) == 0):
            #     print("\n=== Epoch {} ===".format(i))
            #     print("loss: ", avg_loss / num_batches)
        dyn_avg_loss = avg_loss / num_batches
        # logger.log('dynamics model ave_training_loss:%f' %dyn_avg_loss)

        inputs = {k: v for k, v in zip(self.all_phs, self.buf.get())}
        pi_l_old, v_l_old, ent = self.sess.run([self.pi_loss, self.v_loss, self.approx_ent], feed_dict=inputs)

        for i in range(train_pi_iters):
            _, kl = self.sess.run([self.train_pi, self.approx_kl], feed_dict=inputs)
            kl = mpi_avg(kl)
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break
        logger.store(StopIter=i)
        for _ in range(train_v_iters):
            self.sess.run(self.train_v, feed_dict=inputs)

        # Log changes from update
        pi_l_new, v_l_new, kl, cf = self.sess.run([self.pi_loss, self.v_loss, self.approx_kl, self.clipfrac], feed_dict=inputs)

        logger.store(LossPi=pi_l_old, LossV=v_l_old, LossDyn=dyn_avg_loss,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old))
        return training_loss_list, dyn_avg_loss, pi_l_old, v_l_old

    # def save(self, save_dir):
    #     params = OrderedDict()
    #     dyn_list = tf.trainable_variables(scope='dyn')
    #     for var in dyn_list:
    #         params[var.name]= var
    #     pi_list = tf.trainable_variables(scope='pi')
    #     for var in pi_list:
    #         params[var.name]= var
    #     v_list = tf.trainable_variables(scope='v')
    #     for var in v_list:
    #         params[var.name]= var
    #     saver = tf.train.Saver(params)
    #     saver.save(self.sess, save_dir)
    def save(self, save_dir):
        saver = tf.train.Saver()
        saver.save(self.sess, save_dir)

    def restore(self, save_dir):
        saver = tf.train.Saver()
        saver.restore(self.sess, save_dir)

    def run_validation(self, inputs, outputs):

        #init vars
        nData = inputs.shape[0]
        avg_loss=0
        iters_in_batch=0

        for batch in range(int(math.floor(nData / self.batchsize))):
            # Batch the training data
            dataX_batch = inputs[batch*self.batchsize:(batch+1)*self.batchsize, :]
            dataZ_batch = outputs[batch*self.batchsize:(batch+1)*self.batchsize, :]

            #one iteration of feedforward training
            z_predictions, loss = self.sess.run([self.curr_nn_output, self.mse_], feed_dict={self.x_: dataX_batch, self.z_: dataZ_batch})

            avg_loss+= loss
            iters_in_batch+=1

        #avg loss + all predictions
        print ("Validation set size: ", nData)
        print ("Validation set's total loss: ", avg_loss/iters_in_batch)

        return (avg_loss/iters_in_batch)

    def do_forward_sim(self, forwardsim_x_true, forwardsim_y):
        curr_state = np.copy(forwardsim_x_true)
        forwardsim_x_processed = forwardsim_x_true - self.mean_x
        forwardsim_x_processed = np.nan_to_num(forwardsim_x_processed / self.std_x)
        forwardsim_y_processed = forwardsim_y - self.mean_y
        forwardsim_y_processed =np.nan_to_num(forwardsim_y_processed / self.std_y)
        # input = np.concatenate((forwardsim_x_true, forwardsim_y), axis=0)
        input = np.expand_dims(np.append(forwardsim_x_processed, forwardsim_y_processed), axis=0)
        # run through NN to get prediction
        model_output = self.sess.run([self.curr_nn_output], feed_dict={self.x_: input})

        # multiply by std and add mean back in
        state_differences = (model_output[0][:,:-1].flatten() * self.std_z) + self.mean_z
        reward = (model_output[0][:,-1:].flatten() * self.std_r) + self.mean_r

        # update the state info
        next_state = curr_state + state_differences
        if(next_state.min()>50):
            terminal = True
        else:
            terminal = False
        return next_state, reward, terminal

    def update_mean_std(self, mean_x, mean_y, mean_z, mean_r, std_x, std_y, std_z, std_r):
        self.mean_x = mean_x
        self.std_x = std_x
        self.mean_y = mean_y
        self.std_y = std_y
        self.mean_z = mean_z
        self.std_z = std_z
        self.mean_r = mean_r
        self.std_r = std_r
