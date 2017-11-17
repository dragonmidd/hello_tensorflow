# -*- coding: utf-8 -*-

import tensorflow as tf
my_graph = tf.Graph()
with tf.Session(graph=my_graph) as sess:
    x = tf.constant([1,2,3])
    y = tf.constant([4,5,6])
    
    op = tf.add(x, y)
    res = sess.run(fetches=op)
    print(res)