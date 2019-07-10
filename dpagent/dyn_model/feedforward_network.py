
import numpy as np
import tensorflow as tf

def feedforward_network(inputState, inputSize, outputSize, num_fc_layers, depth_fc_layers, tf_datatype, scope):

    #vars
    intermediate_size=depth_fc_layers
    # reuse= False
    # initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf_datatype)
    # fc = tf.contrib.layers.fully_connected

    # make hidden layers
    for i in range(num_fc_layers):
        if(i==0):
            fc_i = tf.layers.dense(inputState, units=intermediate_size, activation=None, name='hidden_'+str(i))
            # fc_i = fc(inputState, num_outputs=intermediate_size, activation_fn=None, weights_initializer=initializer,
            #           biases_initializer=initializer, reuse=reuse, trainable=True, scope=scope+'/hidden_'+str(i))
        else:
            fc_i = tf.layers.dense(h_i, units=intermediate_size, activation=None, name='hidden_' + str(i))
            # fc_i = fc(h_i, num_outputs=intermediate_size, activation_fn=None, weights_initializer=initializer,
            #           biases_initializer=initializer, reuse=reuse, trainable=True, scope=scope+'/hidden_'+str(i))
        h_i = tf.nn.relu(fc_i)

    # make output layer
    z = tf.layers.dense(h_i, units=outputSize, activation=None, name='output')
    # z=fc(h_i, num_outputs=outputSize, activation_fn=None, weights_initializer=initializer,
    #     biases_initializer=initializer, reuse=reuse, trainable=True, scope=scope+'/output')
    print(tf.global_variables())
    return z