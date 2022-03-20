### ### 2022.3.18 Z.Zhang
'''
The code is used for testing the FTCNN network to measure the linewidths of gold lines

Network input: 1.TSOM image, MAT format

2.Focused image, MAT format

Network output: linewidths

'''

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import os
import h5py
import time
import numpy as np
import pandas as pd

from sklearn import preprocessing
os.environ["CUDA_VISIBLE_DEVICES"]="0"

time_start = time.time()
np.random.seed(1)

# tensorflow V1.x
tf.set_random_seed(1234)

dir="D:\ZZG\小离焦测试集/"

xTsom_Dis0 = h5py.File(dir + "CTsomPic.mat",'r')
xTsom_Dis1 = xTsom_Dis0['CTsomPic'][:]
xTsom_Dis = np.transpose(xTsom_Dis1)
xTsom_Dis = pd.DataFrame(xTsom_Dis)
#xTsom_Dis = sio.loadmat(dir + "CTsomPic.mat")
#xTsom_Dis = pd.DataFrame(xTsom_Dis["CTsomPic"])

xFocus_Dis0 = h5py.File(dir + "CFocusPic.mat",'r')
xFocus_Dis1 = xFocus_Dis0['CFocusPic'][:]
xFocus_Dis = np.transpose(xFocus_Dis1)
xFocus_Dis = pd.DataFrame(xFocus_Dis)
# xFocus_Dis = sio.loadmat(dir + "CDeFocusPic97.mat")
# xFocus_Dis = pd.DataFrame(xFocus_Dis["CDeFocusPic97"])

y_Dis0 = h5py.File(dir + "CLabel.mat",'r')
y_Dis1 = y_Dis0['CLabel'][:]
y_Dis = np.transpose(y_Dis1)
#y = sio.loadmat(dir + "CLabel.mat")
y_Dis = pd.DataFrame(y_Dis)

test_xTsom_Dis= np.array(xTsom_Dis)
test_xFocus_Dis= np.array(xFocus_Dis)
test_y_Dis = np.array(y_Dis)

#x_MinMax = preprocessing.MinMaxScaler()

ss_x = preprocessing.StandardScaler()
test_xFocus_Dis = ss_x.fit_transform(test_xFocus_Dis)
test_xTsom_Dis = ss_x.fit_transform(test_xTsom_Dis)

# test_xTsom_Dis= test_xTsom_Dis[0:6800,:]
# test_xFocus_Dis = test_xFocus_Dis[0:6800,:]
# test_y_Dis = test_y_Dis[0:6800,:]

# ss_x = preprocessing.StandardScaler()
# train_xTsom_dis = ss_x.fit_transform(train_xTsom_dis)
# train_xFocus_dis = ss_x.fit_transform(train_xFocus_dis)

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

#xs = tf.placeholder(tf.float32, [None, 7921])
xsTsom = tf.placeholder(tf.float32, [None,7921])
xsFocus = tf.placeholder(tf.float32, [None,7921])
ys = tf.placeholder(tf.float32, [None, 1])

keep_prob = tf.placeholder(tf.float32)

#xTsom_image=tf.reshape(xs, [-1, 89, 89, 1])
xTsom_image = tf.reshape(xsTsom,[-1,89,89,1])
xFocus_image = tf.reshape(xsFocus,[-1,89,89,1])
#############   TSOM image convolution processing    ############

##channel 1##
#####################3*3 kernels###################

## conv1 layer
W_conv1 = weight_variable([3, 3, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(xTsom_image, W_conv1) + b_conv1)
h_pool1=max_pool_2x2(h_conv1)
#print(h_pool1)

## conv2 layer
W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2=max_pool_2x2(h_conv2)
# print(h_pool2)

## conv2 layer

W_conv3 = weight_variable([3, 3, 64, 96])
b_conv3 = bias_variable([96])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
#h_pool3=max_pool_2x2(h_conv3)
print(h_conv3)


##channel 2##
#########################################5*5 kernels########################

## conv1 layer
W_conv15 = weight_variable([5, 5, 1, 64])
b_conv15 = bias_variable([64])
h_conv15 = tf.nn.relu(conv2d2(xTsom_image, W_conv15) + b_conv15)

h_pool15=max_pool_2x2(h_conv15)

# ## conv2 layer
#
W_conv25 = weight_variable([5, 5, 64, 96])
b_conv25 = bias_variable([96])
h_conv25 = tf.nn.relu(conv2d2(h_pool15, W_conv25) + b_conv25)
h_pool25=max_pool_2x2(h_conv25)

## conv2 layer
# #
print(h_pool25)

#########################################################################################################
## conv1 layer

W_conv17 = weight_variable([7, 7, 1, 64])

b_conv17 = bias_variable([64])

h_conv17 = tf.nn.relu(conv2d2(xTsom_image, W_conv17) + b_conv17)
h_pool17=max_pool_2x2(h_conv17)

# ## conv2 layer
#
W_conv27 = weight_variable([7, 7, 64, 96])

b_conv27 = bias_variable([96])

h_conv27 = tf.nn.relu(conv2d2(h_pool17, W_conv27) + b_conv27)
h_pool27=max_pool_2x2(h_conv27)

## conv2 layer
print(h_pool27)

###Convolution processing of Focus image###
W_Fconv1 = weight_variable([3, 3, 1, 32])
b_Fconv1 = bias_variable([32])
h_Fconv1 = tf.nn.relu(conv2d(xFocus_image, W_Fconv1) + b_Fconv1)
h_Fpool1=max_pool_2x2(h_Fconv1)

## conv2 layer
W_Fconv2 = weight_variable([3, 3, 32, 64])
b_Fconv2 = bias_variable([64])
h_Fconv2 = tf.nn.relu(conv2d(h_pool1, W_Fconv2) + b_Fconv2)
h_Fpool2=max_pool_2x2(h_conv2)

## conv2 layer

W_Fconv3 = weight_variable([3, 3, 64, 96])
b_Fconv3 = bias_variable([96])
h_Fconv3 = tf.nn.relu(conv2d(h_Fpool2, W_Fconv3) + b_Fconv3)


###########################################################################################################

con_fe=tf.concat([h_pool27, h_pool25], axis=3)

print(con_fe)

padding_h = (h_conv3.shape)[1] - (con_fe.shape)[1]
padding_w = (h_conv3.shape)[2] - (con_fe.shape)[2]

upsampling = tf.pad(con_fe, ((0, 0), (0, padding_h), (0, padding_w), (0, 0)), 'constant')
con_feature = tf.image.resize(h_conv3, ((upsampling.shape)[1], (upsampling.shape)[2]),
                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
concat_feature = tf.concat([con_feature, upsampling], axis=3)
TF_feature = tf.concat([concat_feature, h_Fconv3], axis=3)
print(concat_feature)
print(TF_feature)
h_pool2_flat = tf.reshape(TF_feature, [-1, 23 * 23 *384])

###The fully connection layer###
W_fc1 = weight_variable([23 * 23 * 384, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 1])
b_fc2 = bias_variable([1])

prediction= tf.matmul(h_fc1_drop, W_fc2) + b_fc2
mse = tf.losses.mean_squared_error(ys, prediction)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    module_file = tf.train.latest_checkpoint('D:\ZZG\FT_Net\模型\model3/')
    saver.restore(sess, module_file)
    ypretaction = np.array(sess.run(prediction, feed_dict={xsTsom: test_xTsom_Dis, xsFocus:test_xFocus_Dis, keep_prob: 1}))
    mse1 = np.array(sess.run(mse, feed_dict={xsTsom: test_xTsom_Dis, xsFocus:test_xFocus_Dis, ys:test_y_Dis, keep_prob: 1}))
    print('mse:',mse1)
    #print("w1:", sess.run(v3))
    #print("b1:", sess.run(v4))
   # print("preya:", ypretaction)
    np.set_printoptions(threshold=np.inf)
  # s = str(test_y_disorder)
    s = str(np.hstack((test_y_Dis, ypretaction)))
    #s = str(ypretaction)
    f = open('D:/ZZG/测试结果/Pianyi1.txt', 'w')
    f.writelines(s)
    f.close()
