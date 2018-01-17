# -*- coding: utf-8 -*-
#TensorFlow实现最近邻算法
#次案例的前提是了解mnist数据集（手写数字识别）
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
#导入mnist数据集
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
mnist = input_data.read_data_sets("d:\data", one_hot=True)
 
#5000样本作为训练集 每一个训练和测试样本的数据都是1*784的矩阵，标签是1*10的矩阵并且采用one-hot编码
X_train , Y_train = mnist.train.next_batch(5000)
#600样本作为测试集
X_test , Y_test = mnist.test.next_batch(200)
 
#创建占位符 None代表将来可以选多个样本的，如：[60,784]代表选取60个样本，每一个样本的是784列
x_train = tf.placeholder("float",[None,784])
x_test = tf.placeholder("float",[784])#x_test代表只用一个样本
#计算距离
#tf.negative(-2)的输出的结果是2
#tf.negative(2)的输出的结果是-2
#reduce_sum的参数reduction_indices解释见下图
#计算一个测试样本和训练样本的的距离
#distance 返回的是N个训练样本的和单个测试样本的距离
distance = tf.reduce_sum(tf.abs(tf.add(x_train,tf.negative(x_test))),reduction_indices=1)
#的到距离最短的训练样本的索引
prediction = tf.arg_min(distance,0)
accuracy = 0
#初始化变量
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
 
    for i in range(len(X_test)):#遍历整个测试集，每次用一个的测试样本和整个训练样本的做距离运算
        #获得最近邻
        # 获得训练集中与本次参与运算的测试样本最近的样本编号
        nn_index = sess.run(prediction,feed_dict={x_train:X_train,x_test:X_test[i,:]})
        #打印样本编号的预测类别和准确类别
        print("Test",i,"Prediction:",np.argmax(Y_train[nn_index]),"True Class:",np.argmax(Y_test[i]))
        if np.argmax(Y_train[nn_index]) == np.argmax(Y_test[i]):
            #如果预测正确。更新准确率
            accuracy += 1./len(X_test)
    print("完成！")
    print("准确率：",accuracy)
