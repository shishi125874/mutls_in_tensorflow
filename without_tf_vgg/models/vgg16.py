# coding=utf-8
import inspect
import os

import numpy as np
import tensorflow as tf
import time

VGG_MEAN = [103.939, 116.779, 123.68]


# kw,kh卷积核大小,
def conv(input_tensor, name, kw, kh, n_out, dw=1, dh=1, activation_fn=tf.nn.relu):
    n_in = input_tensor.get_shape()[-1].value
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', [kh, kw, n_in, n_out], tf.float32,
                                  tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(
            "bias", [n_out], tf.float32, tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input_tensor, weights,
                            (1, dh, dw, 1), padding='SAME')
        activation = activation_fn(tf.nn.bias_add(conv, biases))
        return activation


def fully_connected(input_tensor, name, n_out, activation_fn=tf.nn.relu):
    n_in = input_tensor.get_shape()[-1].value
    with tf.variable_scope(name):
        weights = tf.get_variable(
            'weights', [n_in, n_out], tf.float32, tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(
            "bias", [n_out], tf.float32, tf.constant_initializer(0.0))
        logits = tf.nn.bias_add(tf.matmul(input_tensor, weights), biases)
        return activation_fn(logits)


def pool(input_tensor, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_tensor,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='VALID',
                          name=name)


def loss(logits, onehot_labels):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(
        logits, onehot_labels, name='xentropy')
    loss = tf.reduce_mean(xentropy, name='loss')
    return loss


def topK_error(predictions, labels, K=5):
    correct = tf.cast(tf.nn.in_top_k(predictions, labels, K), tf.float32)
    accuracy = tf.reduce_mean(correct)
    error = 1.0 - accuracy
    return error


def average_gradients(grads):
    average_grads = []
    for grad_and_vars in zip(*grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def build(input_tensor, n_classes=1000, rgb_mean=None, training=True):
    # assuming 224x224x3 input_tensor

    # define image mean
    if rgb_mean is None:
        rgb_mean = np.array([116.779, 123.68, 103.939], dtype=np.float32)
    mu = tf.constant(rgb_mean, name="rgb_mean")
    keep_prob = 0.5

    # subtract image mean
    net = tf.subtract(input_tensor, mu, name="input_mean_centered")

    # block 1 -- outputs 112x112x64
    net = conv(net, name="conv1_1", kh=3, kw=3, n_out=64)
    net = conv(net, name="conv1_2", kh=3, kw=3, n_out=64)
    net = pool(net, name="pool1", kh=2, kw=2, dw=2, dh=2)

    # block 2 -- outputs 56x56x128
    net = conv(net, name="conv2_1", kh=3, kw=3, n_out=128)
    net = conv(net, name="conv2_2", kh=3, kw=3, n_out=128)
    net = pool(net, name="pool2", kh=2, kw=2, dh=2, dw=2)

    # # block 3 -- outputs 28x28x256
    net = conv(net, name="conv3_1", kh=3, kw=3, n_out=256)
    net = conv(net, name="conv3_2", kh=3, kw=3, n_out=256)
    net = pool(net, name="pool3", kh=2, kw=2, dh=2, dw=2)

    # block 4 -- outputs 14x14x512
    net = conv(net, name="conv4_1", kh=3, kw=3, n_out=512)
    net = conv(net, name="conv4_2", kh=3, kw=3, n_out=512)
    net = conv(net, name="conv4_3", kh=3, kw=3, n_out=512)
    net = pool(net, name="pool4", kh=2, kw=2, dh=2, dw=2)

    # block 5 -- outputs 7x7x512
    net = conv(net, name="conv5_1", kh=3, kw=3, n_out=512)
    net = conv(net, name="conv5_2", kh=3, kw=3, n_out=512)
    net = conv(net, name="conv5_3", kh=3, kw=3, n_out=512)
    net = pool(net, name="pool5", kh=2, kw=2, dw=2, dh=2)

    # flatten
    flattened_shape = np.prod([s.value for s in net.get_shape()[1:]])
    net = tf.reshape(net, [-1, flattened_shape], name="flatten")

    # fully connected
    net = fully_connected(net, name="fc6", n_out=4096)
    net = tf.nn.dropout(net, keep_prob)
    net = fully_connected(net, name="fc7", n_out=4096)
    net = tf.nn.dropout(net, keep_prob)
    net = fully_connected(net, name="fc8", n_out=n_classes)

    return net


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        # 计算参数的均值，并使用tf.summary.scaler记录
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        # 计算参数的标准差
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        # 用直方图记录参数的分布
        tf.summary.histogram('histogram', var)
