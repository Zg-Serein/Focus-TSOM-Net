### 2022.3.18 Z.Zhang
'''
The code is used for the construction and training of FTCNN network to measure the linewidths of gold lines

Network input: 1.TSOM image, MAT format

2.Focused image, MAT format

Network output: linewidths

'''

import tensorflow.compat.v1 as tf


tf.disable_v2_behavior()

import os
import time
import h5py
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

os.environ["CUDA_VISIBLE_DEVICES"]="0"

time_start = time.time()
np.random.seed(1)

# tensorflow V1.x
tf.set_random_seed(1234)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Load the data set, xTsom represents TSOM image data, xFocus represents focus position related image, y represents label
## If mat file is saved as V-7.3 or above, use H5PY, otherwise use the loadmat

dir="D:\ZZG\TSOM数据集/双输入v3/"
#xTsom_Dis = sio.loadmat(dir + "CTsomPic.mat")
#xFocus_Dis = sio.loadmat(dir + "CFocusPic.mat")
#y_Dis=sio.loadmat(dir + "CLabel.mat")

xTsom_Dis0 = h5py.File(dir + "CTsomPic.mat",'r')
xFocus_Dis0 = h5py.File(dir + "CFocusPic.mat",'r')
y_Dis0 = h5py.File(dir + "CLabel.mat",'r')

xTsom_Dis1 = xTsom_Dis0['CTsomPic'][:]
xFocus_Dis1 = xFocus_Dis0['CFocusPic'][:]
y_Dis1 = y_Dis0['CLabel'][:]

xTsom_Dis = np.transpose(xTsom_Dis1)
xFocus_Dis = np.transpose(xFocus_Dis1)
y_Dis = np.transpose(y_Dis1)

xTsom_Dis = pd.DataFrame(xTsom_Dis)
xFocus_Dis = pd.DataFrame(xFocus_Dis)

# Two data sets were merged, randomly divided into training set and validation set, and then split
# This is to ensure a one-to-one correspondence between the two inputs

x_Dis = pd.concat([xTsom_Dis, xFocus_Dis],axis=1,ignore_index= True)
y_Dis = pd.DataFrame(y_Dis)
train_x_Dis, val_x_Dis, train_y_Dis, val_y_Dis = train_test_split(x_Dis, y_Dis, test_size=0.2, random_state=42)

## split
#train_xTsom_Dis = train_x_Dis.loc["CTsomPic"]
#train_xFocus_Dis = train_x_Dis.loc["CFocusPic"]
#test_xTsom_Dis = test_x_Dis.loc["CTsomPic"]
#test_xFocus_Dis = test_x_Dis.loc["CFocusPic"]
train_xTsom_Dis =  train_x_Dis.iloc[:,0:7921]
train_xFocus_Dis = train_x_Dis.iloc[:,7921:15842]
val_xTsom_Dis =  val_x_Dis.iloc[:,0:7921]
val_xFocus_Dis = val_x_Dis.iloc[:,7921:15842]

train_xTsom_Dis = np.array(train_xTsom_Dis)
train_xFocus_Dis = np.array(train_xFocus_Dis)
train_y_Dis = np.array(train_y_Dis)
val_xTsom_Dis = np.array(val_xTsom_Dis)
val_xFocus_Dis = np.array(val_xFocus_Dis)
val_y_Dis = np.array(val_y_Dis)

val_xTsom_Dis = val_xTsom_Dis[0:6800,:]
val_xFocus_Dis = val_xFocus_Dis[0:6800,:]
val_y_Dis = val_y_Dis[0:6800,:]

#trainTsom1 = trainTsom[1]

## show
#img1 = np.reshape(train_xTsom_Dis[11:12], (89, 89))
#sc1= plt.imshow(img1)
#plt.show()

## Disrupt the sequence of training data
shuffle_index = np.random.permutation(np.arange(len(train_x_Dis)))
#train_x_Dis=train_x_Dis[shuffle_index]
train_xTsom_Dis = train_xTsom_Dis[shuffle_index]
train_xFocus_Dis = train_xFocus_Dis[shuffle_index]
train_y_Dis = train_y_Dis[shuffle_index]

#standardized
ss_x = preprocessing.StandardScaler()
train_xFocus_Dis = ss_x.fit_transform(train_xFocus_Dis)
train_xTsom_Dis = ss_x.fit_transform(train_xTsom_Dis)
val_xFocus_Dis = ss_x.transform(val_xFocus_Dis)
val_xTsom_Dis = ss_x.transform(val_xTsom_Dis)

'''
################################################
'''

batch_size = 16   # Using MBGD, set batch_size

def generatebatch(X1,X2,Y,n_examples, batch_size):
    for batch_i in range(n_examples // batch_size):
        start = batch_i*batch_size
        end = start + batch_size
        batch_xs1 = X1[start:end]
        batch_xs2 = X2[start:end]
        batch_ys = Y[start:end]
        yield batch_xs1, batch_xs2, batch_ys

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2d2(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

def conv2d3(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 3, 3, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# define placeholder for inputs to network

xsTsom = tf.placeholder(tf.float32, [None,7921])
xsFocus = tf.placeholder(tf.float32, [None,7921])
ys = tf.placeholder(tf.float32, [None, 1])

keep_prob = tf.placeholder(tf.float32)


xTsom_image = tf.reshape(xsTsom,[-1,89,89,1])
xFocus_image = tf.reshape(xsFocus,[-1,89,89,1])
#############   TSOM image convolution processing    ############

##channel1##
#####################3*3 kernels###################

## conv1 layer
W_conv1 = weight_variable([3, 3, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(xTsom_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
#print(h_pool1)

## conv2 layer
W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# print(h_pool2)

## conv2 layer
W_conv3 = weight_variable([3, 3, 64, 96])
b_conv3 = bias_variable([96])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
#h_pool3=max_pool_2x2(h_conv3)
print(h_conv3)


##channel2##
#########################################5*5 kernels########################

## conv1 layer
W_conv15 = weight_variable([5, 5, 1, 64])
b_conv15 = bias_variable([64])
h_conv15 = tf.nn.relu(conv2d2(xTsom_image, W_conv15) + b_conv15)
h_pool15=max_pool_2x2(h_conv15)
# #print(h_pool1)

##conv2 layer
W_conv25 = weight_variable([5, 5, 64, 96])
b_conv25 = bias_variable([96])
h_conv25 = tf.nn.relu(conv2d2(h_pool15, W_conv25) + b_conv25)
h_pool25=max_pool_2x2(h_conv25)
print(h_pool25)

###########################################7*7 kernel######################
## conv1 layer
W_conv17 = weight_variable([7, 7, 1, 64])
b_conv17 = bias_variable([64])
h_conv17 = tf.nn.relu(conv2d2(xTsom_image, W_conv17) + b_conv17)
h_pool17=max_pool_2x2(h_conv17)

## conv2 layer
W_conv27 = weight_variable([7, 7, 64, 96])
b_conv27 = bias_variable([96])
h_conv27 = tf.nn.relu(conv2d2(h_pool17, W_conv27) + b_conv27)
h_pool27=max_pool_2x2(h_conv27)
###h_pool27=max_pool_2x2(h_conv27)
print(h_pool27)

###Convolution processing of Focus image###
W_Fconv1 = weight_variable([3, 3, 1, 32])
b_Fconv1 = bias_variable([32])
h_Fconv1 = tf.nn.relu(conv2d(xFocus_image, W_Fconv1) + b_Fconv1)
h_Fpool1=max_pool_2x2(h_Fconv1)
#print(h_pool1)

##conv2 layer
W_Fconv2 = weight_variable([3, 3, 32, 64])
b_Fconv2 = bias_variable([64])
h_Fconv2 = tf.nn.relu(conv2d(h_pool1, W_Fconv2) + b_Fconv2)
h_Fpool2=max_pool_2x2(h_conv2)

##conv2 layer

W_Fconv3 = weight_variable([3, 3, 64, 96])
b_Fconv3 = bias_variable([96])
h_Fconv3 = tf.nn.relu(conv2d(h_Fpool2, W_Fconv3) + b_Fconv3)


###########################################################################################################
###### preprocessing before Full-connection layer ######
con_fe=tf.concat([h_pool27, h_pool25], axis=3)
print(con_fe)

padding_h = (h_conv3.shape)[1] - (con_fe.shape)[1]
padding_w = (h_conv3.shape)[2] - (con_fe.shape)[2]

upsampling = tf.pad(con_fe, ((0, 0), (0, padding_h), (0, padding_w), (0, 0)), 'constant')
con_feature = tf.image.resize(h_conv3, ((upsampling.shape)[1], (upsampling.shape)[2]),
                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # Crop the feature graph that needs to be spliced
concat_feature = tf.concat([con_feature, upsampling], axis=3)  # Splicing characteristic maps of extended and contracted layers
TF_feature = tf.concat([concat_feature, h_Fconv3], axis=3)
print(concat_feature)
print(TF_feature)
h_pool2_flat = tf.reshape(TF_feature, [-1, 23 * 23 *384])


###The fully connection layer###
W_fc1 = weight_variable([23 * 23 * 384, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
## fc2 layer
W_fc2 = weight_variable([1024, 1])
b_fc2 = bias_variable([1])

prediction= tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Calculate the predition and Y gap. Methods used: suare() square,sum(),mean()
cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
R2S = 1-(tf.reduce_sum(tf.square(ys-prediction))/tf.reduce_sum(tf.square(ys-tf.reduce_mean(ys))))
mse = tf.losses.mean_squared_error(ys, prediction)

Mape = tf.reduce_mean(tf.reduce_sum(tf.abs((ys - prediction)/ys), reduction_indices=[1]))*10000
Mae = tf.reduce_mean(tf.reduce_sum(tf.abs(ys - prediction), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

sess = tf.Session()

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(2000):
        #batch_i = 0
        for batch_xs1,batch_xs2,batch_ys in generatebatch(train_xTsom_Dis, train_xFocus_Dis, train_y_Dis, train_y_Dis.shape[0], batch_size): # 每个周期进行MBGD算法
            sess.run(train_step,feed_dict={xsTsom:batch_xs1,xsFocus:batch_xs2,ys:batch_ys, keep_prob: 1})

        if(epoch%1==0):
            time_end = time.time()

            Msetrain = sess.run(cross_entropy,feed_dict={xsTsom:batch_xs1,xsFocus:batch_xs2,ys:batch_ys, keep_prob: 1})
            Maetrain = sess.run(Mae,feed_dict={xsTsom:batch_xs1,xsFocus:batch_xs2,ys:batch_ys, keep_prob: 1})

            Msetest = sess.run(cross_entropy, feed_dict={xsTsom: val_xTsom_Dis, xsFocus: val_xFocus_Dis, ys: val_y_Dis, keep_prob: 1})
            Maetest = sess.run(Mae, feed_dict={xsTsom: val_xTsom_Dis, xsFocus: val_xFocus_Dis, ys: val_y_Dis, keep_prob: 1})
            #MAPEtest = sess.run(Mape, feed_dict={xsTsom: test_xTsom_Dis, xsFocus: test_xFocus_Dis, ys: test_y_Dis, keep_prob: 1})
            # R2Stest=sess.run(R2S, feed_dict={xsTsom: test_xTsom_Dis, xsFocus:test_xFocus_Dis,ys: test_y_Dis, keep_prob: 1})
            # ypretaction = np.array(sess.run(prediction,feed_dict={xsTsom: test_xTsom_Dis, xsFocus: test_xFocus_Dis, ys: test_y_Dis,
            #                                             keep_prob: 1}))
            # ypretactiondis = np.array(sess.run(prediction, feed_dict={xs: disorder, keep_prob: 1}))
            if (Msetest <= 3):
                saver.save(sess, "D:\ZZG\FT_Net\模型\model5/FtNet_model")
                np.set_printoptions(threshold=np.inf)
                # s = str(np.hstack((test_y_Dis, ypretaction)))
                # f = open('D:/ZZG/测试结果/Output4.txt', 'w')
                # f.writelines(s)
                # f.close()

            print (epoch,'Training Mse=',Msetrain,'Training Mae=',Maetrain,'Testing Mse=',Msetest,'Testing Mae=',Maetest,
            'time=',time_end - time_start,
            flush=True)


# ----------------------------------Visualization of each layer feature-------------------------------
    # imput image
    np.set_printoptions(threshold=np.inf)
    fig2= plt.subplots(figsize=(2, 2))
    img=np.reshape(train_xTsom_Dis[11:12], (94, 94))
    print(img)
    sc=plt.imshow(img)
    plt.show()
#
# # The feature graph of the convolution output of the first layer
    input_Tsom = train_xTsom_Dis[11:12]
    conv1_16 = sess.run(h_conv1, feed_dict={xsTsom: input_Tsom})  # [1, 28, 28 ,16]
    conv1_transpose = sess.run(tf.transpose(conv1_16, [3, 0, 1, 2]))
    fig3, ax3 = plt.subplots(nrows=1, ncols=32, figsize=(64, 2))
    for i in range(32):
       ax3[i].imshow(conv1_transpose[i][0])  # tensor的切片[row, column]

    plt.title('Tsom Conv1 16x28x28')
    plt.show()

# The feature graph of the pooling output of the first layer
    pool1_16 = sess.run(h_pool1, feed_dict={xsTsom: input_Tsom})  # [1, 14, 14, 16]
    pool1_transpose = sess.run(tf.transpose(pool1_16, [3, 0, 1, 2]))
    fig4, ax4 = plt.subplots(nrows=1, ncols=32, figsize=(32, 1))
    for i in range(32):
      ax4[i].imshow(pool1_transpose[i][0])

    plt.title('Tsom Pool1 16x14x14')
    plt.show()

 # The feature graph of the convolution output of the second layer
    conv2_32 = sess.run(h_conv2, feed_dict={xsTsom: input_Tsom})  # [1, 14, 14, 32]
    conv2_transpose = sess.run(tf.transpose(conv2_32, [3, 0, 1, 2]))
    fig5, ax5 = plt.subplots(nrows=1, ncols=64, figsize=(64, 1))
    for i in range(64):
       ax5[i].imshow(conv2_transpose[i][0])
    plt.title('Tsom Conv2 32x14x14')
    plt.show()

#  The feature graph of the pooling output of the second layer
    pool2_32 = sess.run(h_pool2, feed_dict={xsTsom: input_Tsom})  # [1, 7, 7, 32]
    pool2_transpose = sess.run(tf.transpose(pool2_32, [3, 0, 1, 2]))
    fig6, ax6 = plt.subplots(nrows=1, ncols=64, figsize=(64, 1))
    plt.title('Tsom Pool2 32x7x7')
    for i in range(64):
      ax6[i].imshow(pool2_transpose[i][0])
    plt.show()


#Focus图像的特征输出
    #The feature graph of the convolution output of the first layer

    input_Focus = train_xFocus_Dis[11:12]
    Fconv1_16 = sess.run(h_Fconv1, feed_dict={xsTsom: input_Focus})  # [1, 28, 28 ,16]
    Fconv1_transpose = sess.run(tf.transpose(Fconv1_16, [3, 0, 1, 2]))
    fig3, ax3 = plt.subplots(nrows=1, ncols=32, figsize=(64, 2))
    for i in range(32):
       ax3[i].imshow(conv1_transpose[i][0])  # tensor的切片[row, column]

    plt.title('Focus Conv1 16x28x28')
    plt.show()

    # The feature graph of the pooling output of the first layer
    pool1_16 = sess.run(h_pool1, feed_dict={xsTsom: input_Focus})  # [1, 14, 14, 16]
    pool1_transpose = sess.run(tf.transpose(pool1_16, [3, 0, 1, 2]))
    fig4, ax4 = plt.subplots(nrows=1, ncols=32, figsize=(32, 1))
    for i in range(32):
      ax4[i].imshow(pool1_transpose[i][0])

    plt.title('Focus Pool1 16x14x14')
    plt.show()



