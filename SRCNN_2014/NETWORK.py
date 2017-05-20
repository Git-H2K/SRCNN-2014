import tensorflow as tf
import numpy as np

def model(input_tensor):
    with tf.device("/gpu:0"):
        Weight = []

        # First layer
        conv_w_0 = tf.get_variable("conv_w_0", [9, 9, 1, 64],initializer=tf.random_normal_initializer(stddev=0.001))
        conv_b_0 = tf.get_variable("conv_b_0", [64], initializer=tf.constant_initializer(0))
        # Save weights
        Weight.append(conv_w_0)
        Weight.append(conv_b_0)
        Output = tf.nn.relu((tf.nn.conv2d(input_tensor, conv_w_0, strides=[1, 1, 1, 1], padding='VALID')+conv_b_0))

        # Second layer
        conv_w_1 = tf.get_variable("conv_w_1", [1, 1, 64, 32],initializer=tf.random_normal_initializer(stddev=0.001))
        conv_b_1 = tf.get_variable("conv_b_1", [32], initializer=tf.constant_initializer(0))
        # Save weights
        Weight.append(conv_w_1)
        Weight.append(conv_b_1)
        Output = tf.nn.relu((tf.nn.conv2d(Output, conv_w_1, strides=[1, 1, 1, 1], padding='VALID')+conv_b_1))

        # Third layer
        conv_w_2 = tf.get_variable("conv_w_2", [5, 5, 32, 1],initializer=tf.random_normal_initializer(stddev=0.001))
        conv_b_2 = tf.get_variable("conv_b_2", [1], initializer=tf.constant_initializer(0))
        # Save weights
        Weight.append(conv_w_2)
        Weight.append(conv_b_2)
        Output = (tf.nn.conv2d(Output, conv_w_2, strides=[1, 1, 1, 1], padding='VALID')+conv_b_2)

        return Output, Weight
