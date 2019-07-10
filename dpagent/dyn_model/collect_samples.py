import numpy as np
import time
import matplotlib.pyplot as plt
import copy

class CollectSamples(object):

    def __init__(self, env, policy):
        self.env = env
        self.policy = policy

        self.low = self.env.observation_space.low
        self.high = self.env.observation_space.high
        self.shape = self.env.observation_space.shape

        self.use_low = self.low + (self.high-self.low)/3.0
        self.use_high = self.high - (self.high-self.low)/3.0

    def collect_samples(self, num_rollouts, steps_per_rollout):
        observations_list = []
        actions_list = []
        rewards_list = []
        rolloutrewards_list = []
        visualization_frequency = 10
        for rollout_number in range(num_rollouts):
            observation= self.env.reset()
            observations, actions, rewards, reward_for_rollout = self.perform_rollout(observation, steps_per_rollout,
                                                                        rollout_number, visualization_frequency)

            rolloutrewards_list.append(reward_for_rollout)
            observations= np.array(observations)
            actions= np.array(actions)
            observations_list.append(observations)
            actions_list.append(actions)
            rewards_list.append(rewards)

        #return list of length = num rollouts
        #each entry of that list contains one rollout
        #each entry is [steps_per_rollout x statespace_dim] or [steps_per_rollout x actionspace_dim]
        return observations_list, actions_list, rewards_list, rolloutrewards_list

    def perform_rollout(self, observation, steps_per_rollout, rollout_number, visualization_frequency):
        observations = []
        actions = []
        rewards = []
        # visualize = False
        reward_for_rollout = 0
        if((rollout_number%visualization_frequency)==0):
            print("currently performing rollout #", rollout_number)

        for step_num in range(steps_per_rollout):
            action, _ = self.policy.get_action(observation)

            observations.append(observation)
            actions.append(action)

            next_observation, reward, terminal, _ = self.env.step(action)
            rewards.append(reward)
            reward_for_rollout+= reward

            observation = np.copy(next_observation)
            
            if terminal:
                print("Had to stop rollout because terminal state was reached.")
                break

        return observations, actions, rewards, reward_for_rollout

    def perform_step(self, observation, action):
        next_observation, reward, terminal, _ = self.env.step(action)
        return next_observation, reward, terminal