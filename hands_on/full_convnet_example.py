import tensorflow as tf
import numpy as np


x = np.random.normal(0,1,(1000,32,32,3))
y = np.random.choice(30,1000)
y_onehot = tf.one_hot(
    y,
    30,
    on_value=0.99999
)

# network
X = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
Y = tf.placeholder(tf.float32, shape=(None, 30))

conv0 = tf.layers.conv2d(
    X,
    filters=64,
    kernel_size=3,
    padding="same",
    name="conv2d/0",
    activation=tf.nn.elu)
pool0 = tf.layers.max_pooling2d(
    conv0, pool_size=2, strides=2, padding="same")
conv1 = tf.layers.conv2d(
    pool0,
    filters=64,
    kernel_size=3,
    padding="same",
    name="conv2d/1",
    activation=tf.nn.elu)
pool1 = tf.layers.max_pooling2d(
    conv1, pool_size=2, strides=2, padding="same")
conv2 = tf.layers.conv2d(
    pool1,
    filters=64,
    kernel_size=3,
    padding="same",
    name="conv2d/2",
    activation=tf.nn.elu)
flat0 = tf.layers.flatten(conv2)
dense0 = tf.layers.dense(flat0,30)

out = tf.nn.sigmoid(dense0)

loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(Y, out))
optimizer = tf.train.AdamOptimizer()
optimize = optimizer.minimize(loss)

epochs = 100
batch_size = 16

with tf.Session() as sess:
    sess.run(tf.initializers.global_variables())
    for epoch in range(epochs):
        for i in range(0,len(x),batch_size):
            if i+batch_size-1<len(x):
                x_batch = x[i:i+batch_size]
                y_batch = y_onehot[i:i+batch_size]
                print(x_batch.shape)
                print(y_batch.shape)

        # result = sess.run(optimze, feed_dict={X: x_batch, Y: y_batch})
    