# coding=utf-8
import glob
import os
import tensorflow as tf
import numpy as np
import time
import cv2
from models import vgg16


# 将所有的图片resize成100*100
# 224, 224
# w = 100
# h = 100
w = 224
h = 224
c = 3

data_path = '/media/_Data/multil_classification/tfrecord'
tf_data_list = []
test_data_list = []
data_classes = os.listdir(data_path)
nor_path = len(data_classes)
train_num = int(nor_path*0.8)
for i in range(nor_path):
    tf_path = os.path.join(data_path, data_classes[i])
    if i < train_num:
        tf_data_list.append(tf_path)
    else:
        test_data_list.append(tf_path)


filename_queue = tf.train.string_input_producer(tf_data_list)  # 读入流中
test_filename_queue = tf.train.string_input_producer(test_data_list)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
_, serialized_test = reader.read(test_filename_queue)
features = tf.parse_single_example(serialized_example,
                                   features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'img_raw': tf.FixedLenFeature([], tf.string),
                                   })  # 取出包含image和label的feature对象

test_features = tf.parse_single_example(serialized_example,
                                        features={
                                            'label': tf.FixedLenFeature([], tf.int64),
                                            'img_raw': tf.FixedLenFeature([], tf.string),
                                        })  # 取出包含image和label的feature对象

img = tf.decode_raw(features['img_raw'], tf.uint8)
img = tf.reshape(img, [224, 224, 3])
img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
label = tf.cast(features['label'], tf.int32)

test_img = tf.decode_raw(features['img_raw'], tf.uint8)
test_img = tf.reshape(test_img, [224, 224, 3])
test_img = tf.cast(test_img, tf.float32) * (1. / 255) - 0.5
label = tf.cast(test_features['label'], tf.int32)


# 组合batch
batch_size = 200
mini_after_dequeue = 100
capacity = mini_after_dequeue+3*batch_size
# reader = tf.TFRecordReader()

example_batch, label_batch = tf.train.shuffle_batch(
    [img, label], batch_size=batch_size, capacity=capacity)

test_batch, test_label_batch = tf.train.batch(
    [test_img, test_label], batch_size=batch_size, capacity=capacity)

# -----------------构建网络----------------------
# 占位符
x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')
# train_mode = tf.placeholder(tf.bool)

# vgg = vgg16.Vgg16()
logits = vgg16.build(x)
# print(vgg16.get_var_count())

# ---------------------------网络结束---------------------------

# 学习速率指数递减
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(
    0.001, global_step, 100, decay_rate=0.98, staircase=True)
tf.name_scope('loss'):
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=logits)
    tf.summary.scalar('loss',loss)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
with tf.name_scope('acc'):
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('acc',loss)

merged = tf.summary.merge_all()
saver = tf.train.Saver()

# 训练和测试数据，可将n_epoch设置更大一些

n_epoch = 10
with tf.Session as sess:
    writer = tf.summary.FileWriter("/media/_Data/multil_classification/logs/", sess.graph)
    sess.run(tf.global_variables_initializer())
    threads = tf.train.start_queue_runners(sess=sess)
    for epoch in range(n_epoch):
        start_time = time.time()
        print epoch

        # training
        train_loss, train_acc, n_batch = 0, 0, 0
        for i in range(328000/batch_size):
            x_train, y_train = sess.run([example_batch, label_batch])
            _, err, ac = sess.run([train_op, loss, acc], feed_dict={
                                  x: x_train, y_: y_train})
            print('epoch[%d] in [%d / %d] : Loss %f  acc %f' %
                  (epoch, n_batch, i, err, ac))
            tf.scalar_summary('accuracy', ac)
            tf.scalar_summary('error', err)

        train_ac = (train_acc / n_batch)

        # validation
        val_loss, val_acc, n_batch = 0, 0, 0
        for i in range(82000/batch_size):
            x_test, y_test = sess.run([test_batch, test_label_batch])
            err, ac = sess.run([loss, acc], feed_dict={
                               x: x_test, y_: y_test})
            val_loss += err
            val_acc += ac
            n_batch += 1
        print("   validation loss: %f" % (val_loss / n_batch))
        print("   validation acc: %f" % (val_acc / n_batch))
        val_ac = val_acc / n_batch
        tf.scalar_summary('test_val_ac', val_ac)

        with open("acc.txt", "a") as f:
            f.write("train acc" + str(epoch) + ":" + str(train_ac) + '\n')
            f.write("validation acc" + str(epoch) + ":" + str(val_ac) + '\n')

        if epoch % 10 == 0:
            saver.save(sess, "/home/shi/project/mutils/Model/model",
                       global_step=epoch)
