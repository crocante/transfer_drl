
import numpy as np
import numpy.random as npr
import tensorflow as tf
import time
import math

from dpagent.dyn_model.feedforward_network import feedforward_network


class Dyn_Model:

    def __init__(self, inputSize, outputSize, sess, learning_rate, batchsize,
                num_fc_layers, depth_fc_layers, mean_x=0, mean_y=0, mean_z=0, mean_r=0,
                 std_x=0, std_y=0, std_z=0, std_r=0, tf_datatype=tf.float32):

        #init vars
        self.sess = sess
        self.batchsize = batchsize
#         self.which_agent = which_agent
#         self.x_index = x_index
#         self.y_index = y_index
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.mean_x = mean_x
        self.mean_y = mean_y
        self.mean_z = mean_z
        self.mean_r = mean_r
        self.std_x = std_x
        self.std_y = std_y
        self.std_z = std_z
        self.std_r = std_r
#         self.print_minimal = print_minimal

        #placeholders
        self.x_ = tf.placeholder(tf_datatype, shape=[None, self.inputSize], name='x') #inputs
        self.z_ = tf.placeholder(tf_datatype, shape=[None, self.outputSize], name='z') #labels

        #forward pass
        with tf.variable_scope('dyn'):
            self.curr_nn_output = feedforward_network(self.x_, self.inputSize, self.outputSize,
                                                    num_fc_layers, depth_fc_layers, tf_datatype)

        self.dyn_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dyn')
        #loss
        self.mse_ = tf.reduce_mean(tf.square(self.z_ - self.curr_nn_output))

        # Compute gradients and update parameters
        self.opt = tf.train.AdamOptimizer(learning_rate)
        self.theta = tf.trainable_variables()
        self.gv = [(g,v) for g,v in
                    self.opt.compute_gradients(self.mse_, self.theta)
                    if g is not None]
        self.train_step = self.opt.apply_gradients(self.gv)

    def train(self, dataX, dataZ, nEpoch, save_dir):

        #init vars
        start = time.time()
        training_loss_list = []
        range_of_indeces = np.arange(dataX.shape[0])
        nData = dataX.shape[0]

        batchsize = int(self.batchsize)

        #training loop
        for i in range(nEpoch):
            
            #reset to 0
            avg_loss=0
            num_batches=0

            #randomly order indeces (equivalent to shuffling dataX and dataZ)
            indeces = npr.choice(range_of_indeces, size=(dataX.shape[0],), replace=False)

            #get through the full old dataset
            for batch in range(int(math.floor(nData / batchsize))):
                #walk through the randomly reordered "old data"
                dataX_batch = dataX[indeces[batch*batchsize:(batch+1)*batchsize], :]
                dataZ_batch = dataZ[indeces[batch*batchsize:(batch+1)*batchsize], :]

                #one iteration of feedforward training
                _, loss, output, true_output = self.sess.run([self.train_step, self.mse_, self.curr_nn_output, self.z_],
                                                            feed_dict={self.x_: dataX_batch, self.z_: dataZ_batch})
                training_loss_list.append(loss)
                avg_loss+= loss
                num_batches+=1

            #save losses after an epoch
            np.save(save_dir + '/training_losses.npy', training_loss_list)
            if((i%10)==0):
                print("\n=== Epoch {} ===".format(i))
                print ("loss: ", avg_loss/num_batches)
#             if(not(self.print_minimal)):
#                 if((i%10)==0):
#                     print("\n=== Epoch {} ===".format(i))
#                     print ("loss: ", avg_loss/num_batches)
        
#         if(not(self.print_minimal)):
        print ("Training set size: ", (batchsize))
        print("Training duration: {:0.2f} s".format(time.time()-start))

        return (avg_loss/num_batches)

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

    #multistep prediction using the learned dynamics model at each step
    def do_forward_sim(self, forwardsim_x_true, forwardsim_y, many_in_parallel):

        #init vars
        state_list = []

        if(many_in_parallel):
            #init vars
            N= forwardsim_y.shape[0]
            horizon = forwardsim_y.shape[1]
            array_stdz = np.tile(np.expand_dims(self.std_z, axis=0),(N,1))
            array_meanz = np.tile(np.expand_dims(self.mean_z, axis=0),(N,1))
            array_stdy = np.tile(np.expand_dims(self.std_y, axis=0),(N,1))
            array_meany = np.tile(np.expand_dims(self.mean_y, axis=0),(N,1))
            array_stdx = np.tile(np.expand_dims(self.std_x, axis=0),(N,1))
            array_meanx = np.tile(np.expand_dims(self.mean_x, axis=0),(N,1))

            if(len(forwardsim_x_true)==2):
                #N starting states, one for each of the simultaneous sims
                curr_states=np.tile(forwardsim_x_true[0], (N,1))
            else:
                curr_states=np.copy(forwardsim_x_true)

            #advance all N sims, one timestep at a time
            for timestep in range(horizon):

                #keep track of states for all N sims
                state_list.append(np.copy(curr_states))

                #make [N x (state,action)] array to pass into NN
                states_preprocessed = np.nan_to_num(np.divide((curr_states-array_meanx), array_stdx))
                actions_preprocessed = np.nan_to_num(np.divide((forwardsim_y[:,timestep,:]-array_meany), array_stdy))
                inputs_list= np.concatenate((states_preprocessed, actions_preprocessed), axis=1)

                #run the N sims all at once
                model_output = self.sess.run([self.curr_nn_output], feed_dict={self.x_: inputs_list})
                state_differences = np.multiply(model_output[0][:,:-1],array_stdz)+array_meanz

                #update the state info
                curr_states = curr_states + state_differences

            #return a list of length = horizon+1... each one has N entries, where each entry is (13,)
            state_list.append(np.copy(curr_states))
        else:
            curr_state = np.copy(forwardsim_x_true[0]) #curr state is of dim NN input

            for curr_control in forwardsim_y:

                state_list.append(np.copy(curr_state))
                # curr_control = np.expand_dims(curr_control, axis=0)

                #subtract mean and divide by standard deviation
                curr_state_preprocessed = curr_state - self.mean_x
                curr_state_preprocessed = np.nan_to_num(curr_state_preprocessed/self.std_x)
                curr_control_preprocessed = curr_control - self.mean_y
                curr_control_preprocessed = np.nan_to_num(curr_control_preprocessed/self.std_y)
                inputs_preprocessed = np.concatenate((curr_state_preprocessed,curr_control_preprocessed), axis=1)
                # inputs_preprocessed = np.expand_dims(np.append(curr_state_preprocessed, curr_control_preprocessed), axis=0)

                #run through NN to get prediction
                model_output = self.sess.run([self.curr_nn_output], feed_dict={self.x_: inputs_preprocessed}) 

                #multiply by std and add mean back in
                state_differences= (model_output[0][0]*self.std_z)+self.mean_z

                #update the state info
                next_state = curr_state + state_differences

                #copy the state info
                curr_state= np.copy(next_state)

            state_list.append(np.copy(curr_state))

        return state_list

    def do_forward(self, forwardsim_x_true, forwardsim_y):
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
        state_differences = (model_output[0][:,:-1] * self.std_z) + self.mean_z
        reward = (model_output[0][:,-1:] * self.std_r) + self.mean_r

        # update the state info
        next_state = curr_state + state_differences
        return next_state, reward