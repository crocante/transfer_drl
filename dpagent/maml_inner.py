import tensorflow as tf
import numpy as np
from collections import OrderedDict
from tt_utils.utils import remove_scope_from_name
from dpagent.mlp import create_mlp, forward_mlp

def build_policy(self, obs_dim, act_dim):
    self.obs_var, self.mean_var = create_mlp(name='mean_network',
                                             output_dim=act_dim,
                                             hidden_sizes=self.hidden_sizes,
                                             hidden_nonlinearity=self.hidden_nonlinearity,
                                             output_nonlinearity=self.output_nonlinearity,
                                             input_dim=(None, obs_dim,)
                                             )
    with tf.variable_scope("log_std_network"):
        log_std_var = tf.get_variable(name='log_std_var',
                                      shape=(1, act_dim,),
                                      dtype=tf.float32,
                                      initializer=tf.constant_initializer(self.init_log_std),
                                      trainable=True
                                      )
        self.log_std_var = tf.maximum(log_std_var, np.log(1e-6), name='log_std')
    self.action_var = self.mean_var + tf.random_normal(shape=tf.shape(self.mean_var)) * tf.exp(log_std_var)
    # save the policy's trainable variables in dicts
    current_scope = tf.get_default_graph().get_name_scope()
    trainable_policy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=current_scope)
    self.policy_params = OrderedDict(
        [(remove_scope_from_name(var.name, current_scope), var) for var in trainable_policy_vars])
    # self.mean_network_params = [x for x in tf.trainable_variables() if 'mean_network' in x.name]
    # self.log_std_network_params = [x for x in tf.trainable_variables() if 'log_std_var' in x.name]


def build_graph(self, learning_rate, obs_dim, act_dim):
    with tf.variable_scope("ph_graph"):
        with tf.variable_scope("mean_network"):
            # create mean network parameter placeholders
            mean_network_phs = self._create_placeholders_for_vars(
                scope="mean_network")  # -> returns ordered dict
            # forward pass through the mean mpl
            _, mean_var = forward_mlp(output_dim=act_dim,
                                      hidden_sizes=self.hidden_sizes,
                                      hidden_nonlinearity=self.hidden_nonlinearity,
                                      output_nonlinearity=self.output_nonlinearity,
                                      input_var=self.obs_var,
                                      mlp_params=mean_network_phs,
                                      )
        with tf.variable_scope("log_std_network"):
            # create log_stf parameter placeholders
            log_std_network_phs = self._create_placeholders_for_vars(
                scope="log_std_network")  # -> returns ordered dict
            log_std_var = list(log_std_network_phs.values())[0]  # weird stuff since log_std_network_phs is ordered dict
        action_var = mean_var + tf.random_normal(shape=tf.shape(mean_var)) * tf.exp(log_std_var)
        # merge mean_network_phs and log_std_network_phs into policies_params_phs
        self.policies_params_phs = []
        odict = mean_network_phs
        odict.update(mean_network_phs)
        self.policies_params_phs.append(odict)
        self.policy_params_keys = list(self.policies_params_phs[0].keys())

    with tf.variable_scope('dyn_update'):
        self.dyn_vars = [x for x in tf.trainable_variables() if 'dyn' in x.name]
        self.mse_ = tf.reduce_mean(tf.square(self.z_ - self.curr_nn_output))
        self.opt = tf.train.AdamOptimizer(learning_rate)
        self.gv = [(g, v) for g, v in
                   self.opt.compute_gradients(self.mse_, self.dyn_vars)
                   if g is not None]
        self.train_step = self.opt.apply_gradients(self.gv)

    with tf.variable_scope('policy_update'):
        mean_network_params = OrderedDict()
        log_std_network_params = []
        for name, param in self.policies_params_phs[0].items():
            if 'log_std_network' in name:
                log_std_network_params.append(param)
            else:  # if 'mean_network' in name:
                mean_network_params[name] = param

        obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, obs_dim], name='obs')
        action_ph = tf.placeholder(dtype=tf.float32, shape=[None, act_dim], name='action')
        adv_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='advantage')
        obs_ph, mean_var = forward_mlp(output_dim=act_dim,
                                       hidden_sizes=self.hidden_sizes,
                                       hidden_nonlinearity=self.hidden_nonlinearity,
                                       output_nonlinearity=self.output_nonlinearity,
                                       input_var=obs_ph,
                                       mlp_params=mean_network_params,
                                       )

        log_std_var = log_std_network_params[0]
        # distribution_info_new = dict(mean=mean_var, log_std=log_std_var)
        # means = distribution_info_new["mean"]
        # log_stds = distribution_info_new["log_std"]
        with tf.variable_scope("log_likelihood"):
            zs = (self.action_var - mean_var) / tf.exp(log_std_var)
            log_likelihood_adapt = - tf.reduce_sum(log_std_var, reduction_indices=-1) - \
                                   0.5 * tf.reduce_sum(tf.square(zs), reduction_indices=-1) - \
                                   0.5 * act_dim * np.log(2 * np.pi)
        with tf.variable_scope("surrogate_loss"):
            surr_obj_adapt = -tf.reduce_mean(log_likelihood_adapt * adv_ph)
        with tf.variable_scope("adapt_step"):
            update_param_keys = list(self.policies_params_phs[0].keys())
            grads = tf.gradients(surr_obj_adapt, [self.policies_params_phs[0][key] for key in update_param_keys])
            gradients = dict(zip(update_param_keys, grads))
            # gradient descent
            adapted_policy_params = [self.policies_params_phs[0][key] - tf.multiply(self.step_size, gradients[key])
                                     for key in update_param_keys]
            adapted_policy_params_dict = OrderedDict(zip(update_param_keys, adapted_policy_params))