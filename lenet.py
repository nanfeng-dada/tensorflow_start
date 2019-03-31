import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # 下载并加载mnist数据
x = tf.placeholder(tf.float32, [None, 784])  # 输入的数据占位符
y_actual = tf.placeholder(tf.float32, shape=[None, 10])  # 输出的标签占位符

# 第一层卷积池化：28*28*1---》14*14*6
x_image=tf.reshape(x,[-1,28,28,1])
w_conv1=tf.Variable(tf.truncated_normal([5,5,1,6], stddev=0.1))
b_conv1=tf.Variable(tf.constant(0.1,shape=[6]))
h_conv1=tf.nn.relu(tf.nn.conv2d(input=x_image, filter=w_conv1, strides=[1, 1, 1, 1], padding='SAME'))
h_pool1=tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 第二层卷积池化：14*14*6--》7*7*16
w_conv2=tf.Variable(tf.truncated_normal([5,5,6,16], stddev=0.1))
b_conv2=tf.Variable(tf.constant(0.1,shape=[16]))
h_conv2=tf.nn.relu(tf.nn.conv2d(input=h_pool1, filter=w_conv2, strides=[1, 1, 1, 1], padding='SAME'))
h_pool2=tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 将池化层输出展开为1-D，方便全连接层
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*16])

#三个全连接层
w_fc1=tf.Variable(tf.truncated_normal([7*7*16,120], stddev=0.1))
b_fc1=tf.Variable(tf.constant(0.1,shape=[120]))
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)

w_fc2=tf.Variable(tf.truncated_normal([120,84], stddev=0.1))
b_fc2=tf.Variable(tf.constant(0.1,shape=[84]))
h_fc2=tf.nn.relu(tf.matmul(h_fc1,w_fc2)+b_fc2)

w_fc3=tf.Variable(tf.truncated_normal([84,10], stddev=0.1))
b_fc3=tf.Variable(tf.constant(0.1,shape=[10]))
h_fc3=tf.nn.softmax(tf.matmul(h_fc2,w_fc3)+b_fc3)

# 定义代价函数和优化方法
cross_entropy = -tf.reduce_sum(y_actual*tf.log(h_fc3))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(h_fc3, 1), tf.argmax(y_actual, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
for i in range(1000):
    batch = mnist.train.next_batch(60)
    if i%100 == 0:
        train_accuracy = accuracy.eval(session=sess, feed_dict={x: batch[0], y_actual: batch[1]})
        print('step {}, training accuracy: {}'.format(i, train_accuracy))
    train_step.run(session=sess, feed_dict={x: batch[0], y_actual: batch[1]})

print('test accuracy: {}'.format(accuracy.eval(session=sess, feed_dict={x: mnist.test.images, y_actual:
    mnist.test.labels})))
