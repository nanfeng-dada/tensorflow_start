# TensorFlow中文手册mnist入门
"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import
import gzip
import os
import tempfile
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


IMAGE_SIZE=28
NUM_CHANNELS=1
max_accuracy=0
step_num=200

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

def plot_an_image(filename):
    num_images=100
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        # data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE * IMAGE_SIZE)

        image = data[np.random.randint(0, 99), :]
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.matshow(image.reshape((IMAGE_SIZE, IMAGE_SIZE)), cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))  # just get rid of ticks
        plt.yticks(np.array([]))
    return image

def disp_img(filename,row,col):
  num_images=100
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    # data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE*IMAGE_SIZE)

    sample_idx = np.random.choice(np.arange(data.shape[0]), row*col)
    sample_images = data[sample_idx, :]
    fig, ax_array = plt.subplots(nrows=row, ncols=col, sharey=True, sharex=True, figsize=(col, row))

    for r in range(row):
      for c in range(col):
        ax_array[r, c].matshow(sample_images[col * r + c].reshape((IMAGE_SIZE, IMAGE_SIZE)),
                               cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    return sample_images

def load_model():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('model/*.meta')
        saver.restore(sess, tf.train.latest_checkpoint("model/"))






if __name__ == '__main__':

    #导入数据
    # 这里，mnist是一个轻量级的类。它以Numpy数组的形式存储着训练、校验和测试数据集。
    # 同时提供了一个函数，用于在迭代中获得minibatch，后面我们将会用到。
    mnist = read_data_sets("data/", one_hot=True)
    if os._exists('model'):
        del_file('model')
    #显示图片
    disp_img('data/train-images-idx3-ubyte.gz',4,4)
    plt.show()


    # 构建神经BP神经网络


    # 占位符placeholder，可输入任意第一维数
    with tf.name_scope('x'):
        x = tf.placeholder("float", [None, 784])
    W = tf.Variable(tf.zeros([784,10]),name='W')
    b = tf.Variable(tf.zeros([10]),name='b')

    # y=softmax(x*W+b)-----实际计算出的y
    with tf.name_scope('y'):
        y = tf.nn.softmax(tf.matmul(x,W) + b)

    # 标签中的y
    y_ = tf.placeholder("float", [None,10])

    # 代价函数---交叉熵，
    # ‘*’代表笛卡尔积，tf.reduce_sum():计算张量各维度元素和
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))

    # 优化算法：梯度下降，学习率/搜索步长=0.01
    # 还有其他优化算法，后续尝试
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    saver = tf.train.Saver(max_to_keep=1)

    # 训练1000次---随机梯度下降
    for i in range(step_num):

      #每次训练，抓取100个数据进行批量梯度下降
      # TensorFlow中文教程中说是随机抓取，具体需要看mnist.py中的初始化的方法
        batch_xs, batch_ys = mnist.train.next_batch(100)

        step=sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        # 操作返回值none
        # print(step)

    #   tf.argmax(y, axis=1) 返回标签y的最大值的索引下标，由于最大值为1.所以会返回多个下标
    # tf.equal(x,y):x,y对应元素相等则返回true，否则为false，返回布尔向量
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    # 将布尔型向量转换为1，0的浮点数，计算均值就是准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        value_acc=sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

        if value_acc >= max_accuracy:
            max_accuracy = value_acc

            saver.save(sess, "model/my-model", global_step=i)
            stor_i=i
            print("---------------------------save the model--------------------")
        # 这是错误范例，因为TensorFlow先构建计算图，再带入数据得到结果，这样打印出来的是一些张量定义的信息
        # 正确的做法是如上，feed_dict={}重新载入计算图中张量的数据
        # print('step{}，accuracy={}'.format(i,accuracy))
        print('step{}:测试集准确率：{:.4f}'.format(i, value_acc))
    print("训练结束--------")



    while(1):
        str = input("按下任意键,进行测试,退出请按q:")
        if str=='q':
            print('程序结束')
            break
        else:
            img_test=plot_an_image('data/t10k-images-idx3-ubyte.gz')
            img_test=img_test.reshape((-1,784))
            plt.show()
            with tf.Session() as sess:
                saver = tf.train.import_meta_graph('model/my-model-{}.meta'.format(stor_i))
                saver.restore(sess, tf.train.latest_checkpoint("model/"))

                ans=np.argmax(sess.run(y, feed_dict={x: img_test}))
                print('预测值为:',ans)


















