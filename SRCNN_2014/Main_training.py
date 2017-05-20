import math
import numpy as np
import tensorflow as tf
from GENERATE_TRAINING_DATA import Generate_training_data
from NETWORK import model

Data_path = "./train/"

Batch_size = 128
Epoch = 10000
Learning_rate_init = 0.00001

if __name__ == '__main__':

    # Generate training patches , LR_input = Training patches , HR_input = Training labels
    LR_input, HR_input = Generate_training_data(Data_path)
    LR_input_size = (np.shape(LR_input))
    HR_input_size = (np.shape(HR_input))

    # Iteration = total training patches / batch size
    Iter = math.floor(LR_input_size[0] / Batch_size)

    Train_img_LR = tf.placeholder(tf.float32,shape=(None,LR_input_size[1],LR_input_size[2],1))
    Train_img_HR = tf.placeholder(tf.float32,shape=(None,HR_input_size[1],HR_input_size[2],1))

    # Network
    Train_output,Weight = model(Train_img_LR)

    # L2-norm
    Loss = tf.reduce_mean(tf.nn.l2_loss(Train_output-Train_img_HR))

    # To save weights
    saver = tf.train.Saver(Weight, max_to_keep=0)
    Global_step = tf.Variable(0, trainable=False)

    # Gradient descent optimizer
    Opt_Init = tf.train.GradientDescentOptimizer(Learning_rate_init).minimize(Loss)

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        for epc in range(Epoch):
            for it in range(Iter):
                # Make batch LR input and HR input
                LR_input_batch = LR_input[Batch_size*it:Batch_size*(it+1),:,:,:]
                HR_input_batch = HR_input[Batch_size*it:Batch_size*(it+1),:,:,:]

                # Run optimization
                sess.run(Opt_Init, feed_dict={Train_img_LR:LR_input_batch, Train_img_HR:HR_input_batch})

            # Display epoch and loss
            acc=sess.run(Loss, feed_dict={Train_img_LR:LR_input_batch, Train_img_HR:HR_input_batch})
            print([epc,acc])

            saver.save(sess, "./save_param/Epoch_%03d" % epc, global_step=Global_step)