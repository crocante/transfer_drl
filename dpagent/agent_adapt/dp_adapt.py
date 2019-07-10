import numpy as np
import tensorflow as tf
import numpy.random as npr
import math
import time
from collections import OrderedDict

import dpagent.agent_adapt.core as core
from dpagent.agent_adapt.mlp import forward_mlp
from dpagent.ppo.ppo import PPOBuffer
# from dpagent.dyn_model.feedforward_network import feedforward_network
# from tt_utils.logx import EpochLogger
from tt_utils.mpi_tf import MpiAdamOptimizer            #, sync_all_params
# from tt_utils.mpi_tools import mpi_avg       #, mpi_fork, proc_id, mpi_statistics_scalar, num_procs

class DP_Adapt():
    def __init__(self, params, inputSize, outputSize, env, local_steps_per_epoch, learning_rate, clip_ratio=0.2,
                 pi_lr=3e-4, vf_lr=1e-3, actor_critic=core.mlp_actor_critic, gamma=0.99, lam=0.97, mean_x=0, mean_y=0,
                 mean_z=0, mean_r=0, std_x=0, std_y=0, std_z=0, std_r=0, tf_datatype=tf.float32):
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        print("initializing dynamics model......")
        # scope.reuse_variables()
        self.x_ = tf.placeholder(tf_datatype, shape=[None, inputSize], name='x')  # inputs
        self.z_ = tf.placeholder(tf_datatype, shape=[None, outputSize], name='z')  # labels
        dyn_network_params = OrderedDict()
        pi_network_params = OrderedDict()
        v_network_params = OrderedDict()
        log_std_network_params = []
        for name, param in params.items():
            if 'dyn' in name:
                dyn_network_params[name] = param
            elif 'pi' in name:
                pi_network_params[name] = param
            else:
                v_network_params[name] = param
        # self.curr_nn_output = feedforward_network(self.x_, inputSize, outputSize, num_fc_layers,
        #                                           depth_fc_layers, tf_datatype, scope='dyn')
        hidden_sizes = (500,)
        hidden_nonlinearity = tf.tanh
        output_nonlinearity = None
        _, self.curr_nn_output = forward_mlp(output_dim=obs_dim+1,
                                  hidden_sizes=hidden_sizes,
                                  hidden_nonlinearity=hidden_nonlinearity,
                                  output_nonlinearity=output_nonlinearity,
                                  input_var=self.x_,
                                  mlp_params=dyn_network_params,
                                  )
        var_counts = tuple(core.count_vars(scope) for scope in ['dyn'])
        print('\nNumber of parameters: \t dyn: %d\n' % var_counts)

        self.x_ph, self.a_ph = core.placeholders_from_spaces(env.observation_space, env.action_space)
        self.adv_ph, self.ret_ph, self.logp_old_ph = core.placeholders(None, None, None)
        self.all_phs = [self.x_ph, self.a_ph, self.adv_ph, self.ret_ph, self.logp_old_ph]
        print("initializing policy model........")
        ac_kwargs = dict()
        ac_kwargs['action_space'] = env.action_space  # Share information about action space with policy architecture
        self.pi, self.logp, self.logp_pi, self.v = actor_critic(self.x_ph, self.a_ph, pi_network_params,
                                                                v_network_params, **ac_kwargs)    # Main outputs from computation graph

        self.buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)    # Experience buffer
        var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])        # Count variables
        print('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)
        self.get_action_ops = [self.pi, self.v, self.logp_pi]        # Every step, get: action, value, and logprob
        # dynamic model
        # self.batchsize = batchsize
        self.mean_x = mean_x
        self.mean_y = mean_y
        self.mean_z = mean_z
        self.mean_r = mean_r
        self.std_x = std_x
        self.std_y = std_y
        self.std_z = std_z
        self.std_r = std_r
        # policy model
        # self.train_pi_iters = train_pi_iters
        # self.train_v_iters = train_v_iters
        # self.epochs = epochs
        self.lam = lam
        self.build_graph(learning_rate, clip_ratio, pi_lr, vf_lr)
        params = tf.trainable_variables()
        print(params)
        # self.max_ep_len = max_ep_len
        # self.target_kl = target_kl
        # self.save_freq = save_freq
        # self.logger = EpochLogger(**logger_kwargs)
        # self.logger.save_config(locals())

    def initialize(self, sess):
        self.sess = sess
        var_list = [var for var in tf.global_variables() if var not in tf.get_default_graph().get_collection("trainable_variables")]
        self.sess.run(tf.variables_initializer(var_list))   #global_variables_initializer
        # self.sess.run(sync_all_params())

    def build_graph(self, dyn_lr, clip_ratio, pi_lr, vf_lr):
        with tf.variable_scope('dyn_update'):
            self.dyn_vars = [x for x in tf.trainable_variables() if 'dyn' in x.name]
            self.mse_ = tf.reduce_mean(tf.square(self.z_ - self.curr_nn_output))
            self.opt = tf.train.AdamOptimizer(dyn_lr)
            self.gv = [(g, v) for g, v in
                       self.opt.compute_gradients(self.mse_, self.dyn_vars)
                       if g is not None]
            self.train_step = self.opt.apply_gradients(self.gv)
        with tf.variable_scope("policy_update"):
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

    def train(self, dataX, dataZ, dataX_new, dataZ_new, batchsize, nEpoch, save_dir, fraction_use_new):

        # init vars
        start = time.time()
        training_loss_list = []
        range_of_indeces = np.arange(dataX.shape[0])
        nData_old = dataX.shape[0]
        num_new_pts = dataX_new.shape[0]

        # how much of new data to use per batch
        if (num_new_pts < (batchsize * fraction_use_new)):
            batchsize_new_pts = num_new_pts  # use all of the new ones
        else:
            batchsize_new_pts = int(batchsize * fraction_use_new)

        # how much of old data to use per batch
        batchsize_old_pts = int(batchsize - batchsize_new_pts)

        # training loop
        for i in range(nEpoch):

            # reset to 0
            avg_loss = 0
            num_batches = 0

            # randomly order indeces (equivalent to shuffling dataX and dataZ)
            old_indeces = npr.choice(range_of_indeces, size=(dataX.shape[0],), replace=False)
            # train from both old and new dataset
            if (batchsize_old_pts > 0):

                # get through the full old dataset
                for batch in range(int(math.floor(nData_old / batchsize_old_pts))):

                    # randomly sample points from new dataset
                    if (num_new_pts == 0):
                        dataX_new_batch = dataX_new
                        dataZ_new_batch = dataZ_new
                    else:
                        new_indeces = npr.randint(0, dataX_new.shape[0], (batchsize_new_pts,))
                        dataX_new_batch = dataX_new[new_indeces, :]
                        dataZ_new_batch = dataZ_new[new_indeces, :]

                    # walk through the randomly reordered "old data"
                    dataX_old_batch = dataX[old_indeces[batch * batchsize_old_pts:(batch + 1) * batchsize_old_pts], :]
                    dataZ_old_batch = dataZ[old_indeces[batch * batchsize_old_pts:(batch + 1) * batchsize_old_pts], :]

                    # combine the old and new data
                    dataX_batch = np.concatenate((dataX_old_batch, dataX_new_batch))
                    dataZ_batch = np.concatenate((dataZ_old_batch, dataZ_new_batch))

                    # one iteration of feedforward training
                    _, loss, output, true_output = self.sess.run(
                        [self.train_step, self.mse_, self.curr_nn_output, self.z_],
                        feed_dict={self.x_: dataX_batch, self.z_: dataZ_batch})
                    training_loss_list.append(loss)
                    avg_loss += loss
                    num_batches += 1

            # train completely from new set
            else:
                for batch in range(int(math.floor(num_new_pts / batchsize_new_pts))):
                    # walk through the shuffled new data
                    dataX_batch = dataX_new[batch * batchsize_new_pts:(batch + 1) * batchsize_new_pts, :]
                    dataZ_batch = dataZ_new[batch * batchsize_new_pts:(batch + 1) * batchsize_new_pts, :]

                    # one iteration of feedforward training
                    _, loss, output, true_output = self.sess.run(
                        [self.train_step, self.mse_, self.curr_nn_output, self.z_],
                        feed_dict={self.x_: dataX_batch, self.z_: dataZ_batch})

                    training_loss_list.append(loss)
                    avg_loss += loss
                    num_batches += 1

                # shuffle new dataset after an epoch (if training only on it)
                p = npr.permutation(dataX_new.shape[0])
                dataX_new = dataX_new[p]
                dataZ_new = dataZ_new[p]

            # save losses after an epoch
            np.save(save_dir + '/training_losses.npy', training_loss_list)
            if ((i % 10) == 0):
                print("\n=== Epoch {} ===".format(i))
                print("loss: ", avg_loss / num_batches)

        print("Training set size: ", (nData_old + dataX_new.shape[0]))
        print("Training duration: {:0.2f} s".format(time.time() - start))

        # get loss of curr model on old dataset
        avg_old_loss = 0
        iters_in_batch = 0
        for batch in range(int(math.floor(nData_old / batchsize))):
            # Batch the training data
            dataX_batch = dataX[batch * batchsize:(batch + 1) * batchsize, :]
            dataZ_batch = dataZ[batch * batchsize:(batch + 1) * batchsize, :]
            # one iteration of feedforward training
            loss, _ = self.sess.run([self.mse_, self.curr_nn_output],
                                    feed_dict={self.x_: dataX_batch, self.z_: dataZ_batch})
            avg_old_loss += loss
            iters_in_batch += 1
        old_loss = avg_old_loss / iters_in_batch

        # get loss of curr model on new dataset
        avg_new_loss = 0
        iters_in_batch = 0
        for batch in range(int(math.floor(dataX_new.shape[0] / batchsize))):
            # Batch the training data
            dataX_batch = dataX_new[batch * batchsize:(batch + 1) * batchsize, :]
            dataZ_batch = dataZ_new[batch * batchsize:(batch + 1) * batchsize, :]
            # one iteration of feedforward training
            loss, _ = self.sess.run([self.mse_, self.curr_nn_output],
                                    feed_dict={self.x_: dataX_batch, self.z_: dataZ_batch})
            avg_new_loss += loss
            iters_in_batch += 1
        if (iters_in_batch == 0):
            new_loss = 0
        else:
            new_loss = avg_new_loss / iters_in_batch

        # done
        return (avg_loss / num_batches), old_loss, new_loss

    def update_dyn(self, dataX, dataZ, nEpoch, batchsize, logger):
        range_of_indeces = np.arange(dataX.shape[0])
        nData = dataX.shape[0]
        training_loss_list = []
        # batchsize = int(self.batchsize)
        # Training dynamic network
        for i in range(nEpoch):
            avg_loss = 0
            num_batches = 0
            indeces = npr.choice(range_of_indeces, size=(dataX.shape[0],), replace=False)
            for batch in range(int(math.floor(nData / batchsize))):
                # walk through the randomly reordered "old data"
                dataX_batch = dataX[indeces[batch * batchsize:(batch + 1) * batchsize], :]
                dataZ_batch = dataZ[indeces[batch * batchsize:(batch + 1) * batchsize], :]

                # one iteration of feedforward training
                _, loss, output, true_output = self.sess.run(
                    [self.train_step, self.mse_, self.curr_nn_output, self.z_],
                    feed_dict={self.x_: dataX_batch, self.z_: dataZ_batch})
                training_loss_list.append(loss)
                avg_loss += loss
                num_batches += 1

            # if ((i % 10) == 0):
            #     print("\n=== Epoch {} ===".format(i))
            #     print("loss: ", avg_loss / num_batches)
        dyn_avg_loss = avg_loss / num_batches
        logger.store(LossDyn=dyn_avg_loss)
        return dyn_avg_loss

    def update_inner_policy(self, train_pi_iters, train_v_iters, target_kl, logger):  # , logger
        # pi_loss_list=[]
        # v_loss_list=[]
        # self.update_mean_std(mean_x, mean_y, mean_z, mean_r, std_x, std_y, std_z, std_r)
        inputs = {k: v for k, v in zip(self.all_phs, self.buf.get())}
        pi_l_old, v_l_old, ent = self.sess.run([self.pi_loss, self.v_loss, self.approx_ent], feed_dict=inputs)
        for i in range(train_pi_iters):
            _, kl = self.sess.run([self.train_pi, self.approx_kl], feed_dict=inputs)
            # kl = mpi_avg(kl)
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break
        logger.store(StopIter=i)
        for _ in range(train_v_iters):
            self.sess.run(self.train_v, feed_dict=inputs)

        # Log changes from update
        pi_l_new, v_l_new, kl, cf = self.sess.run([self.pi_loss, self.v_loss, self.approx_kl, self.clipfrac],
                                                  feed_dict=inputs)

        logger.store(LossPi_model=pi_l_new, LossV_model=v_l_new,
                     KL_model=kl, Entropy_model=ent, ClipFrac_model=cf,
                     DeltaLossPi_model=(pi_l_new - pi_l_old),
                     DeltaLossV_model=(v_l_new - v_l_old))
        return pi_l_new, v_l_new

    def update_outer_policy(self, train_pi_iters, train_v_iters, target_kl, logger):  # , logger
        # pi_loss_list=[]
        # v_loss_list=[]
        # self.update_mean_std(mean_x, mean_y, mean_z, mean_r, std_x, std_y, std_z, std_r)
        inputs = {k: v for k, v in zip(self.all_phs, self.buf.get())}
        pi_l_old, v_l_old, ent = self.sess.run([self.pi_loss, self.v_loss, self.approx_ent], feed_dict=inputs)
        for i in range(train_pi_iters):
            _, kl = self.sess.run([self.train_pi, self.approx_kl], feed_dict=inputs)
            # kl = mpi_avg(kl)
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break
        logger.store(StopIter=i)
        for _ in range(train_v_iters):
            self.sess.run(self.train_v, feed_dict=inputs)

        # Log changes from update
        pi_l_new, v_l_new, kl, cf = self.sess.run([self.pi_loss, self.v_loss, self.approx_kl, self.clipfrac],
                                                  feed_dict=inputs)

        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old))
        return pi_l_old, v_l_old

    def update(self, dataX, dataZ, mean_x, std_x, mean_y, std_y, mean_z, std_z, mean_r, std_r,
               nEpoch, batchsize, train_pi_iters, train_v_iters, target_kl, logger):   #, logger
        # pi_loss_list=[]
        # v_loss_list=[]
        self.update_mean_std(mean_x, mean_y, mean_z, mean_r, std_x, std_y, std_z, std_r)
        training_loss_list = []
        range_of_indeces = np.arange(dataX.shape[0])
        nData = dataX.shape[0]
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
            # kl = mpi_avg(kl)
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

    def save(self, save_dir):
        saver = tf.train.Saver()
        saver.save(self.sess, save_dir)

    def restore(self, save_dir):
        saver = tf.train.Saver()
        saver.restore(self.sess, save_dir)

    def do_forward_sim(self, forwardsim_x_true, forwardsim_y):
        state_list = []
        # reward_list = []
        N = forwardsim_y.shape[0]
        horizon = forwardsim_y.shape[1]
        # array_stdr = np.tile(np.expand_dims(self.std_r, axis=0), 1)
        # array_meanr = np.tile(np.expand_dims(self.mean_r, axis=0), N)
        array_stdz = np.tile(np.expand_dims(self.std_z, axis=0), (N, 1))
        array_meanz = np.tile(np.expand_dims(self.mean_z, axis=0), (N, 1))
        array_stdy = np.tile(np.expand_dims(self.std_y, axis=0), (N, 1))
        array_meany = np.tile(np.expand_dims(self.mean_y, axis=0), (N, 1))
        array_stdx = np.tile(np.expand_dims(self.std_x, axis=0), (N, 1))
        array_meanx = np.tile(np.expand_dims(self.mean_x, axis=0), (N, 1))

        if (len(forwardsim_x_true) == 2):
            # N starting states, one for each of the simultaneous sims
            curr_states = np.tile(forwardsim_x_true[0], (N, 1))
        else:
            curr_states = np.copy(forwardsim_x_true)

        # advance all N sims, one timestep at a time
        for timestep in range(horizon):
            # keep track of states for all N sims
            state_list.append(np.copy(curr_states))

            # make [N x (state,action)] array to pass into NN
            states_preprocessed = np.nan_to_num(np.divide((curr_states - array_meanx), array_stdx))
            actions_preprocessed = np.nan_to_num(np.divide((forwardsim_y[:, timestep, :] - array_meany), array_stdy))
            inputs_list = np.concatenate((states_preprocessed, actions_preprocessed), axis=1)

            # run the N sims all at once
            model_output = self.sess.run([self.curr_nn_output], feed_dict={self.x_: inputs_list})
            state_differences = np.multiply(model_output[0][:,0:-1], array_stdz) + array_meanz
            # reward = np.multiply(model_output[0][:,-1], array_stdr) + array_meanr
            # reward_list.append(reward)

            # update the state info
            curr_states = curr_states + state_differences

        # return a list of length = horizon+1... each one has N entries, where each entry is (13,)
        state_list.append(np.copy(curr_states))
        #
        # curr_state = np.copy(forwardsim_x_true)
        # forwardsim_x_processed = forwardsim_x_true - self.mean_x
        # forwardsim_x_processed = np.nan_to_num(forwardsim_x_processed / self.std_x)
        # forwardsim_y_processed = forwardsim_y - self.mean_y
        # forwardsim_y_processed =np.nan_to_num(forwardsim_y_processed / self.std_y)
        # # input = np.concatenate((forwardsim_x_true, forwardsim_y), axis=0)
        # input = np.expand_dims(np.append(forwardsim_x_processed, forwardsim_y_processed), axis=0)
        # # run through NN to get prediction
        # model_output = self.sess.run([self.curr_nn_output], feed_dict={self.x_: input})
        #
        # # multiply by std and add mean back in
        # state_differences = (model_output[0][:,:-1].flatten() * self.std_z) + self.mean_z
        # reward = (model_output[0][:,-1:].flatten() * self.std_r) + self.mean_r
        #
        # # update the state info
        # next_state = curr_state + state_differences
        # if(next_state.min()>50):
        #     terminal = True
        # else:
        #     terminal = False
        # return next_state, reward, terminal
        return state_list    #, reward_list

    def do_forward(self, forwardsim_x, forwardsim_y):

        curr_states = np.copy(forwardsim_x)

        # make [N x (state,action)] array to pass into NN
        states_preprocessed = np.nan_to_num(np.divide((forwardsim_x - self.mean_x), self.std_x))
        actions_preprocessed = np.nan_to_num(np.divide((forwardsim_y - self.mean_y), self.std_y))
        inputs_list = np.expand_dims(np.concatenate((states_preprocessed, actions_preprocessed), axis=0), axis=0)

        # run the N sims all at once
        model_output = self.sess.run([self.curr_nn_output], feed_dict={self.x_: inputs_list})
        state_differences = np.multiply(model_output[0][0, 0:-1], self.mean_z) + self.std_z
        reward = np.multiply(model_output[0][0, -1], self.mean_r) + self.std_r

        # update the state info
        curr_states = curr_states + state_differences

        return curr_states, reward

    def update_mean_std(self, mean_x, mean_y, mean_z, mean_r, std_x, std_y, std_z, std_r):
        self.mean_x = mean_x
        self.std_x = std_x
        self.mean_y = mean_y
        self.std_y = std_y
        self.mean_z = mean_z
        self.std_z = std_z
        self.mean_r = mean_r
        self.std_r = std_r

