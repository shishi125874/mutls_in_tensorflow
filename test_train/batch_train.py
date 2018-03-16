#coding=utf-8
import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 路径
swd = '\data\show'
filename_queue = tf.train.string_input_producer(["mydata.tfrecords"])  # 读入流中
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
features = tf.parse_single_example(serialized_example,
                                   features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'img_raw': tf.FixedLenFeature([], tf.string),
                                   })  # 取出包含image和label的feature对象
image = tf.decode_raw(features['img_raw'], tf.uint8)
image = tf.reshape(image, [36, 136, 3])
label = tf.cast(features['label'], tf.int32)

# 组合batch
batch_size = 4
mini_after_dequeue = 100
capacity = mini_after_dequeue+3*batch_size

example_batch, label_batch = tf.train.batch(
    [image, label], batch_size=batch_size, capacity=capacity)

with tf.Session() as sess:  # 开始一个会话
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(10):  # 10个batch
        example, l = sess.run([example_batch, label_batch])  # 取出一个batch
        for j in range(batch_size):  # 每个batch内4张图
            sigle_image = Image.fromarray(example[j], 'RGB')
            sigle_label = l[j]
            sigle_image.save(swd+'batch_'+str(i)+'_'+'size' +
                             str(j)+'_'+'Label_'+str(sigle_label)+'.jpg')  # 存下图片
            print(example, l)

    coord.request_stop()
    coord.join(threads)
