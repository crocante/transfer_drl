import numpy as np
import copy

def add_noise(data_inp, noiseToSignal):
    data= copy.deepcopy(data_inp)
    mean_data = np.mean(data, axis = 0)
    std_of_noise = mean_data*noiseToSignal
    for j in range(mean_data.shape[0]):
        if(std_of_noise[j]>0):
            data[:,j] = np.copy(data[:,j]+np.random.normal(0, np.absolute(std_of_noise[j]), (data.shape[0],)))
    return data

def get_model_samples(env, local_steps_per_epoch, sess, agent, logger, max_ep_len):
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    for t in range(local_steps_per_epoch):
        a, v_t, logp_t = sess.run(agent.get_action_ops, feed_dict={agent.x_ph: o.reshape(1, -1)})

        agent.buf.store(o, a, r, v_t, logp_t)
        logger.store(VVals=v_t)
        observation = np.copy(o)
        action = np.copy(a[0])
        next_o, r = agent.do_forward(observation, action)
        ep_ret += r
        ep_len += 1
        o = next_o

        if (t == local_steps_per_epoch - 1):
            last_val = r if d else sess.run(agent.v, feed_dict={agent.x_ph: o.reshape(1, -1)})
            agent.buf.finish_path(last_val)
            logger.store(AvgRew_model=ep_ret/local_steps_per_epoch)
            # ep_return.append(ep_ret)

def get_samples(env, local_steps_per_epoch, sess, agent, logger, max_ep_len):
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    observations = []
    actions = []
    rewards = []
    for t in range(local_steps_per_epoch):
        a, v_t, logp_t = sess.run(agent.get_action_ops, feed_dict={agent.x_ph: o.reshape(1, -1)})

        # save and log
        # agent.buf.store(o, a, r, v_t, logp_t)
        logger.store(VVals=v_t)
        observation = np.copy(o)
        action = np.copy(a[0])
        observations.append(observation)
        actions.append(action)

        next_o, r, d, _ = env.step(a[0])
        rewards.append(r)
        # store o,a,next_o,r

        ep_ret += r
        ep_len += 1
        o = next_o

        terminal = d or (ep_len == max_ep_len)
        if terminal or (t == local_steps_per_epoch - 1):
            if not (terminal):
                print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)
            # if trajectory didn't reach terminal state, bootstrap value target
            # last_val = r if d else sess.run(agent.v, feed_dict={agent.x_ph: o.reshape(1, -1)})
            # agent.buf.finish_path(last_val)
            if terminal:
                # only save EpRet / EpLen if trajectory finished
                logger.store(EpRet=ep_ret, EpLen=ep_len)
            # ep_return.append(ep_ret)
            # o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            return observations, actions, rewards

def get_samples_with_buf(env, local_steps_per_epoch, sess, agent, logger, max_ep_len):
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    observations = []
    actions = []
    rewards = []
    epoch = 0
    ep_return = 0
    total_rewards = 0
    for t in range(local_steps_per_epoch):
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
        # store o,a,next_o,r

        ep_ret += r
        total_rewards += r
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
                epoch+=1
                # only save EpRet / EpLen if trajectory finished
                # logger.store(EpRet=ep_ret, EpLen=ep_len)
                ep_return += ep_ret
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    return ep_return/epoch, total_rewards/local_steps_per_epoch

def perform_rollouts(env, local_steps_per_epoch, sess, agent, logger, max_ep_len):
    observations = []
    actions = []
    rewards = []
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    for t in range(local_steps_per_epoch):
        a, v_t, logp_t = sess.run(agent.get_action_ops, feed_dict={agent.x_ph: o.reshape(1, -1)})

        # save and log
        # agent.buf.store(o, a, r, v_t, logp_t)
        logger.store(VVals=v_t)
        observation = np.copy(o)
        action = np.copy(a[0])
        observations.append(observation)
        actions.append(action)

        next_o, r, d, _ = env.step(a[0])
        rewards.append(r)
        # store o,a,next_o,r

        ep_ret += r
        ep_len += 1
        o = next_o

        terminal = d or (ep_len == max_ep_len)
        if terminal or (t == local_steps_per_epoch - 1):
            if not (terminal):
                print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)
            # if trajectory didn't reach terminal state, bootstrap value target
            # last_val = r if d else sess.run(agent.v, feed_dict={agent.x_ph: o.reshape(1, -1)})
            # agent.buf.finish_path(last_val)
            if terminal:
                # only save EpRet / EpLen if trajectory finished
                logger.store(EpRet=ep_ret, EpLen=ep_len)
            # ep_return.append(ep_ret)
            # o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            return observations, actions, rewards, ep_ret, ep_len

def collect_samples(num_rollouts, env, steps_per_rollout, sess, agent, logger, max_ep_len):
    observations_list = []
    actions_list = []
    rewards_list = []
    return_list = []
    for rollout_number in range(num_rollouts):
        observations, actions, reward_for_rollout, return_for_rollout, len_for_rollout = perform_rollouts(env,
                                                            steps_per_rollout, sess, agent, logger, max_ep_len)
        rewards_list.append(reward_for_rollout)
        observations = np.array(observations)
        actions = np.array(actions)
        observations_list.append(observations)
        actions_list.append(actions)
        return_list.append(return_for_rollout)

    return observations_list, actions_list, rewards_list

def perform_rollouts_comp(env, local_steps_per_epoch, sess, agent, logger, max_ep_len):
    traj_taken = []
    observations = []
    observations_model = []
    actions = []
    rewards = []
    rewards_model = []
    o, r, d, ep_ret, ret_model, ep_len = env.reset(), 0, False, 0, 0, 0
    traj_taken.append(np.array(o))
    for t in range(local_steps_per_epoch):
        a, v_t, logp_t = sess.run(agent.get_action_ops, feed_dict={agent.x_ph: o.reshape(1, -1)})

        # save and log
        # agent.buf.store(o, a, r, v_t, logp_t)
        logger.store(VVals=v_t)
        observation = np.copy(o)
        action = np.copy(a[0])
        observations.append(observation)
        actions.append(action)

        next_o, r, d, _ = env.step(a[0])
        model_next_o, model_r = agent.do_forward(observation, action)
        observations_model.append(model_next_o)
        traj_taken.append(np.array(next_o))
        rewards.append(r)
        rewards_model.append(model_r)
        # store o,a,next_o,r

        ep_ret += r
        ret_model += model_r
        ep_len += 1
        o = next_o

        terminal = d or (ep_len == max_ep_len)
        if terminal or (t == local_steps_per_epoch - 1):
            if not (terminal):
                print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)
            # if trajectory didn't reach terminal state, bootstrap value target
            # last_val = r if d else sess.run(agent.v, feed_dict={agent.x_ph: o.reshape(1, -1)})
            # agent.buf.finish_path(last_val)
            if terminal:
                # only save EpRet / EpLen if trajectory finished
                logger.store(EpRet=ep_ret, EpLen=ep_len)
            # ep_return.append(ep_ret)
            # o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            return observations, observations_model, actions, rewards, rewards_model, ep_ret, ret_model, ep_len, traj_taken,

def collect_samples_comp(num_rollouts, env, steps_per_rollout, sess, agent, logger, max_ep_len):
    traj_list = []
    x_list = []
    z_list = []
    x_model_list = []
    z_model_list = []
    actions_list = []
    rewards_list = []
    rewards_model_list = []
    return_list = []
    return_model_list = []
    for rollout_number in range(num_rollouts):
        observations, observations_model, actions, reward_for_rollout, model_reward_for_rollout, \
             return_for_rollout, model_return_for_rollout, len_for_rollout, traj_taken = perform_rollouts_comp(env,
                                                                steps_per_rollout, sess, agent, logger, max_ep_len)
        observations_x = np.array(observations)[:, 0]
        observations_z = np.array(observations)[:, 1]
        observations_x_model = np.array(observations_model)[:, 0]
        observations_z_model = np.array(observations_model)[:, 1]
        for i in range(len_for_rollout):
            rewards_list.append(reward_for_rollout[i])
            rewards_model_list.append(model_reward_for_rollout[i])
            x_list.append(observations_x[i])
            x_model_list.append(observations_x_model[i])
            z_list.append(observations_z[i])
            z_model_list.append(observations_z_model[i])

        actions = np.array(actions)
        actions_list.append(actions)
        return_list.append(return_for_rollout)
        return_model_list.append(model_return_for_rollout)
        traj_list.append(np.array(traj_taken))

    return x_list, x_model_list, z_list, z_model_list, actions_list, rewards_list, rewards_model_list, return_list,\
           return_model_list, traj_list

def perform_rollouts_comp1(env, local_steps_per_epoch, sess, agent, logger, max_ep_len):
    traj_taken = []
    observations = []
    actions = []
    rewards = []
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    traj_taken.append(np.array(o))
    for t in range(local_steps_per_epoch):
        a, v_t, logp_t = sess.run(agent.get_action_ops, feed_dict={agent.x_ph: o.reshape(1, -1)})

        # save and log
        # agent.buf.store(o, a, r, v_t, logp_t)
        logger.store(VVals=v_t)
        observation = np.copy(o)
        action = np.copy(a[0])
        observations.append(observation)
        actions.append(action)

        next_o, r, d, _ = env.step(a[0])
        traj_taken.append(np.array(next_o))
        rewards.append(r)
        # store o,a,next_o,r

        ep_ret += r
        ep_len += 1
        o = next_o

        terminal = d or (ep_len == max_ep_len)
        if terminal or (t == local_steps_per_epoch - 1):
            if not (terminal):
                print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)
            # if trajectory didn't reach terminal state, bootstrap value target
            # last_val = r if d else sess.run(agent.v, feed_dict={agent.x_ph: o.reshape(1, -1)})
            # agent.buf.finish_path(last_val)
            if terminal:
                # only save EpRet / EpLen if trajectory finished
                logger.store(EpRet=ep_ret, EpLen=ep_len)
            # ep_return.append(ep_ret)
            # o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            return observations, actions, rewards, ep_ret, ep_len, traj_taken,

def collect_samples_comp1(num_rollouts, env, steps_per_rollout, sess, agent, logger, max_ep_len):
    traj_list = []
    x_list = []
    z_list = []
    actions_list = []
    rewards_list = []
    return_list = []
    for rollout_number in range(num_rollouts):
        observations, actions, reward_for_rollout, return_for_rollout, len_for_rollout, traj_taken = perform_rollouts_comp1(env,
                                                                steps_per_rollout, sess, agent, logger, max_ep_len)
        observations_x = np.array(observations)[:, 0]
        observations_z = np.array(observations)[:, 1]
        for i in range(len_for_rollout):
            rewards_list.append(reward_for_rollout[i])
            x_list.append(observations_x[i])
            z_list.append(observations_z[i])

        actions = np.array(actions)
        actions_list.append(actions)
        return_list.append(return_for_rollout)
        traj_list.append(np.array(traj_taken))

    return x_list, z_list, actions_list, rewards_list, return_list, traj_list