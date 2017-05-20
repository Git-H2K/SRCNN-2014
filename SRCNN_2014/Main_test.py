import numpy as np
import tensorflow as tf
from NETWORK import model
from GENERATE_TESTING_DATA import Generate_testing_data
import matplotlib.pyplot as plt

Data_path = "./test/"

if __name__ == '__main__':
    # Read test image
    Height, Width, Bic_Output, Ori_Img = Generate_testing_data(Data_path)

    Train_img_LR = tf.placeholder(tf.float32,shape=(None,Height,Width,1))
    Train_output,Weight = model(Train_img_LR)

    # To load weights
    saver = tf.train.Saver(Weight)
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        # Load weights
        saver.restore(sess, './save_param/Epoch_6000-0')

        # Run CNN
        Out=sess.run(Train_output, feed_dict={Train_img_LR:np.reshape(Bic_Output,[1,Height,Width,1])})

    plt.close('all')
    # Show output image
    plt.figure(1)
    plt.imshow(((np.reshape(Out,[Height-12,Width-12]))),cmap='gray', vmin=0, vmax=1)

    # Show bicubic image
    plt.figure(2)
    plt.imshow(Bic_Output[6:512-6,6:512-6],cmap='gray', vmin=0, vmax=1)

    # Show Original image
    plt.figure(3)
    plt.imshow(Ori_Img[6:512-6,6:512-6],cmap='gray', vmin=0, vmax=1)
    plt.show()

