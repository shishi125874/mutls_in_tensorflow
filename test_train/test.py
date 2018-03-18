# coding=utf-8
import glob
import os
import tensorflow as tf
import time
import vgg16
import sys
import argparse


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--logs_base_dir', type=str,
                        help='Directory where to write event logs.', default='~/logs/facenet')
    parser.add_argument('--data_path', type=str,
                        help='Directory tfrecord dir.', default='/media/_Data/multil_classification/tfrecord')
    parser.add_argument('--learn_rate', type=float,
                        help='learn_rate begin.', default=0.001)
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--input_weight', type=float,
                        help='input net weight.', default=224)
    parser.add_argument('--input_height', type=float,
                        help='input net height.', default=224)
    return parser.parse_args(argv)


# args = parse_arguments(sys.argv[1:])

# 将所有的图片resize成100*100
# 224, 224
w = 224
h = 224
# w = args.input_weight
# h = args.input_height
c = 3

data_path = '/Users/qianleishi/datasets/traindata.tfrecords-000'
learn_rate_create = 0.001
save_model_dir = os.path.join('models/', 'Model')
board_dir = os.path.join('logs/', 'Logs')
if os.path.exists(save_model_dir):
    os.mkdir(save_model_dir)
if os.path.exists(board_dir):
    os.mkdir(board_dir)

# tf_data_list = []
# test_data_list = []
# data_classes = os.listdir(data_path)
# nor_path = len(data_classes)
# train_num = int(nor_path * 0.8)

filename_queue = tf.train.string_input_producer([data_path, ])  # 读入流中
test_filename_queue = tf.train.string_input_producer([data_path, ])
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

_, serialized_test = reader.read(test_filename_queue)

features = tf.parse_single_example(serialized_example,
                                   features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'img_raw': tf.FixedLenFeature([], tf.string),
                                       'img_width': tf.FixedLenFeature([], tf.int64),
                                       'img_height': tf.FixedLenFeature([], tf.int64),
                                   })  # 取出包含image和label的feature对象

test_features = tf.parse_single_example(serialized_example,
                                        features={
                                            'label': tf.FixedLenFeature([], tf.int64),
                                            'img_raw': tf.FixedLenFeature([], tf.string),
                                            'img_width': tf.FixedLenFeature([], tf.int64),
                                            'img_height': tf.FixedLenFeature([], tf.int64),
                                        })  # 取出包含image和label的feature对象

img = tf.decode_raw(features['img_raw'], tf.uint8)
height = tf.cast(features['img_height'], tf.int32)
width = tf.cast(features['img_width'], tf.int32)
img = tf.reshape(img, [height, width, 3])
img = tf.image.resize_images(img, [w, h], method=tf.image.ResizeMethod.AREA)
img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
label = tf.cast(features['label'], tf.int32)

test_img = tf.decode_raw(test_features['img_raw'], tf.uint8)
test_height = tf.cast(test_features['img_height'], tf.int32)
test_width = tf.cast(test_features['img_width'], tf.int32)
test_img = tf.reshape(test_img, [test_height, test_width, 3])
test_img = tf.image.resize_images(test_img, [w, h], method=tf.image.ResizeMethod.AREA)
test_img = tf.cast(test_img, tf.float32) * (1. / 255) - 0.5
test_label = tf.cast(test_features['label'], tf.int32)

# 组合batch
batch_size = 200
mini_after_dequeue = 100
capacity = mini_after_dequeue + 3 * batch_size
# reader = tf.TFRecordReader()

example_batch, label_batch = tf.train.shuffle_batch(
    [img, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=mini_after_dequeue)

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
    learn_rate_create, global_step, 100, decay_rate=0.98, staircase=True)
with tf.name_scope('loss'):
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=logits)
    tf.summary.scalar('loss', loss)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
with tf.name_scope('acc'):
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('acc', loss)

merged = tf.summary.merge_all()
saver = tf.train.Saver()

# 训练和测试数据，可将n_epoch设置更大一些

n_epoch = 10
with tf.Session() as sess:
    writer = tf.summary.FileWriter(board_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
    threads = tf.train.start_queue_runners(sess=sess)
    # training
    train_loss, train_acc, n_batch = 0, 0, 0
    for i in range(400):
        x_train, y_train = sess.run([example_batch, label_batch])
        _, err, ac = sess.run([train_op, loss, acc], feed_dict={
            x: x_train, y_: y_train})
        print('[%d / %d] : Loss %f  acc %f' %
              (n_batch, i, err, ac))
        tf.scalar_summary('accuracy', ac)
        tf.scalar_summary('error', err)

    train_ac = (train_acc / n_batch)

    # validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for i in range(20):
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
        f.write("train acc" + str(i) + ":" + str(train_ac) + '\n')
        f.write("validation acc" + str(i) + ":" + str(val_ac) + '\n')

    if i % 100 == 0:
        saver.save(sess, save_model_dir,
                   global_step=i)
