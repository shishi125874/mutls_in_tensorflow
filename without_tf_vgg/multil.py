# -*- coding: utf-8 -*-

# from skimage import io, transform
import glob
import os
import tensorflow as tf
import numpy as np
import time
import cv2
from models import vgg16

path = '/media/_Data/multil_classification/deleted'
test_path = '/media/_Data/multil_classification/deleted-test'

# 将所有的图片resize成100*100
# 224, 224
# w = 100
# h = 100
w = 224
h = 224
c = 3


# 读取图片
def read_img(path):
    imgs = []
    labels = []
    path_list = os.listdir(path)
    i = 0
    for item in path_list:
        item_list = os.listdir(os.path.join(path, item))
        for img_item in item_list:
            im = os.path.join(path, item, img_item)
            # print('reading the images:%s' % (im))
            img = cv2.imread(im)
            img = cv2.resize(img, (w, h))
            imgs.append(img)
            labels.append(i)
        i += 1
        print('%s is: %d' % (item, i))
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


# 训练数据
x_train, y_train = read_img(path)

# 打乱顺序
num_example = x_train.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
x_train = x_train[arr]
y_train = y_train[arr]

# 测试数据
x_val, y_val = read_img(test_path)

# 打乱顺序
# num_example = x_val.shape[0]
# arr = np.arange(num_example)
# np.random.shuffle(arr)
# x_val = x_val[arr]
# y_val = y_val[arr]

# 将所有数据分为训练集和验证集
# ratio = 0.8
# s = np.int(num_example * ratio)
# x_train = data[:s]
# y_train = label[:s]
# x_val = data[s:]
# y_val = label[s:]

# -----------------构建网络----------------------
# 占位符
x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')
# train_mode = tf.placeholder(tf.bool)

# vgg = vgg16.Vgg16()
logits = vgg16.build(x)
# print(vgg16.get_var_count())

# ---------------------------网络结束---------------------------

loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=logits)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()


# 定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


# 训练和测试数据，可将n_epoch设置更大一些

n_epoch = 100
batch_size = 64
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for epoch in range(n_epoch):
    start_time = time.time()
    print epoch

    # training
    train_loss, train_acc, n_batch = 0, 0, 0
    i = 0
    minieop = len(x_train) / batch_size
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err
        train_acc += ac
        print('epoch[%d] in [%d / %d] : Loss %f  acc %f' % (epoch, n_batch, minieop, err, ac))
        i += 1
        n_batch += 1
    print("   train loss: %f" % (train_loss / n_batch))
    print("   train acc: %f" % (train_acc / n_batch))

    train_ac = (train_acc / n_batch)

    # validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y_: y_val_a})
        val_loss += err
        val_acc += ac
        n_batch += 1
    print("   validation loss: %f" % (val_loss / n_batch))
    print("   validation acc: %f" % (val_acc / n_batch))
    val_ac = val_acc / n_batch

    with open("acc.txt", "a") as f:
        f.write("train acc" + str(epoch) + ":" + str(train_ac) + '\n')
        f.write("validation acc" + str(epoch) + ":" + str(val_ac) + '\n')

    if epoch % 10 == 0:
        saver.save(sess, "/home/shi/project/mutils/Model/model", global_step=epoch)

sess.close()
