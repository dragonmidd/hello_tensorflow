import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy

img = Image.open('szu.jpg')
img_ndarray = numpy.asarray(img, dtype='float32')
print(img_ndarray.shape)
plt.imshow(img_ndarray)
img_ndarray=img_ndarray[:,:,0]
plt.figure()
plt.subplot(221)
plt.imshow(img_ndarray)

w=[[-1.0,-1.0,-1.0],
   [-1.0,9.0,-1.0],
   [-1.0,-1.0,-1.0]]

with tf.Session() as sess:
    img_ndarray=tf.reshape(img_ndarray,[1,180,240,1])
    w=tf.reshape(w,[3,3,1,1])
    img_cov=tf.nn.conv2d(img_ndarray, w, strides=[1, 1, 1, 1], padding='SAME')
    image_data=sess.run(img_cov)
    print(image_data.shape)
    plt.subplot(222)
    plt.imshow(image_data[0,:,:,0])

    img_pool=tf.nn.max_pool(img_ndarray, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                   padding='SAME')
    image_data = sess.run(img_pool)
    plt.subplot(223)
    plt.imshow(image_data[0, :, :, 0])
    plt.subplot(224)
    img_pool = tf.nn.max_pool(img_ndarray, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1],
                              padding='SAME')
    image_data = sess.run(img_pool)
    plt.imshow(image_data[0, :, :, 0])
    plt.show()