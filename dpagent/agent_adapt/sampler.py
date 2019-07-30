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

        if ((t+1) % max_ep_len ==0):
            last_val = r if d else sess.run(agent.v, feed_dict={agent.x_ph: o.reshape(1, -1)})
            agent.buf.finish_path(last_val)
            logger.store(AvgRew_model=ep_ret/local_steps_per_epoch)

            # ep_return.append(ep_ret)

def get_samples(env, steps_per_epoch, sess, agent, logger, max_ep_len):
    #collect samples and store in PPOBuffer, to train dyn_model in next iteration and do outer_policy_update
    #steps_per_epoch: number of steps should collect
    #max_ep_len: max length of each rollout
    observations_list = []
    actions_list = []
    rewards_list = []
    returns_list = []
    observations = []
    actions = []

    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
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
        rewards_list.append(r)
        ep_ret += r
        ep_len += 1
        o = next_o

        terminal = d or (ep_len == max_ep_len)#or:从左到右扫描，返回第一个为真的表达式值，无真值则返回最后一个表达式值
        if terminal or (t == steps_per_epoch - 1):
            if not (terminal):
                print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)
            # if trajectory didn't reach terminal state, bootstrap value target
            last_val = r if d else sess.run(agent.v, feed_dict={agent.x_ph: o.reshape(1, -1)})
            agent.buf.finish_path(last_val)
            if terminal:
                # only save EpRet / EpLen if trajectory finished
                logger.store(EpRet=ep_ret, EpLen=ep_len)
                returns_list.append(ep_ret)
            observations_list.append(np.array(observations))
            actions_list.append(np.array(actions))
            observations = []
            actions = []
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    return observations_list, actions_list, rewards_list, returns_list

def collect_samples(num_rollouts, env, steps_per_rollout, sess, agent, logger, max_ep_len):
    #collect samples and store in TRPOBuffer, to train dyn_model and policy in next iteration
    for rollout_number in range(num_rollouts):
        perform_rollouts(env, steps_per_rollout, sess, agent, logger, max_ep_len)

    memory_size = agent.AdaptBuf.size
    return memory_size

def perform_rollouts(env, local_steps_per_epoch, sess, agent, logger, max_ep_len):
    # observations = []
    # actions = []
    # rewards = []
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    for t in range(local_steps_per_epoch):
        a, v_t, logp_t = sess.run(agent.get_action_ops, feed_dict={agent.x_ph: o.reshape(1, -1)})

        # save and log
        # agent.buf.store(o, a, r, v_t, logp_t)
        logger.store(VVals=v_t)
        observation = np.copy(o)
        action = np.copy(a[0])
        # observations.append(observation)
        # actions.append(action)

        next_o, r, d, _ = env.step(a[0])
        # rewards.append(r)
        # store o,a,next_o,r

        agent.AdaptBuf.store(observation, action, r, next_o, d)
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
            # return observations, actions, rewards, ep_ret, ep_len

def collect_samples_new(num_rollouts, env, steps_per_rollout, sess, logger, agent, max_ep_len):
    ##########for run_train.py to collect data after update policy  2019.7.28###########
    ##########with trpo_buf#########
    observations_list = []
    actions_list = []
    rewards_list = []
    rolloutrewards_list = []
    for rollout_number in range(num_rollouts):
        observations, actions, rewards, reward_for_rollout = perform_rollouts_new(env, steps_per_rollout, sess, logger,
                                                                                  agent, max_ep_len)
        rolloutrewards_list.append(reward_for_rollout)
        observations = np.array(observations)
        actions = np.array(actions)
        observations_list.append(observations)
        actions_list.append(actions)
        rewards_list.append(rewards)

        # return list of length = num rollouts
        # each entry of that list contains one rollout
        # each entry is [steps_per_rollout x statespace_dim] or [steps_per_rollout x actionspace_dim]
    return observations_list, actions_list, rewards_list, rolloutrewards_list

def perform_rollouts_new(env, steps_per_epoch, sess, logger, agent, max_ep_len):
    observations = []
    actions = []
    rewards = []
    reward_for_rollout = 0
    o, reward, d, ep_ret, ret_model, ep_len = env.reset(), 0, False, 0, 0, 0
    for t in range(steps_per_epoch):
        agent_outs = sess.run(agent.get_action_ops, feed_dict={agent.x_ph: o.reshape(1, -1)})
        a, v_t, logp_t, info_t = agent_outs[0][0], agent_outs[1], agent_outs[2], agent_outs[3:]

        # save and log
        agent.buf.store(o, a, reward, v_t, logp_t, info_t)
        logger.store(VVals=v_t)
        # a, v_t, logp_t = sess.run(agent.get_action_ops, feed_dict={agent.x_ph: o.reshape(1, -1)})
        observation = np.copy(o)
        action = np.copy(a)
        observations.append(observation)
        actions.append(action)

        next_observation, reward, terminal, _ = env.step(action)
        # env.render()
        rewards.append(reward)
        reward_for_rollout += reward

        o = np.copy(next_observation)
        ep_ret += reward
        ep_len += 1

        terminal = d or (ep_len == max_ep_len)
        if terminal or (t == steps_per_epoch - 1):
            if not (terminal):
                print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)
            # if trajectory didn't reach terminal state, bootstrap value target
            last_val = reward if d else sess.run(agent.v, feed_dict={agent.x_ph: o.reshape(1, -1)})
            agent.buf.finish_path(last_val)
            if terminal:
                # only save EpRet / EpLen if trajectory finished
                logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, reward, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    return observations, actions, rewards, reward_for_rollout

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

def get_reward(observation, action, next_observation):
    lb = -np.ones(action.shape, dtype="float32")
    ub = np.ones(action.shape, dtype="float32")
    _normalization_scale = 10.0
    scaled_action = lb + (action + _normalization_scale) * (ub - lb) / (2 * _normalization_scale)
    scaled_action = np.clip(scaled_action, lb, ub)
    reward_ctrl = - 0.5 * 0.1 * np.square(scaled_action).sum()
    reward_run = (next_observation[0] - observation[0])/0.05
    reward = reward_ctrl + reward_run
    return reward