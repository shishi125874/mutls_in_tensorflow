# coding=utf-8
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# 图片地址
swd = ''
# TFRecord文件路径
data_path = '/Users/qianleishi/datasets/traindata.tfrecords-000'
# 获取文件名列表
data_files = tf.gfile.Glob(data_path)

print(data_files)

# 文件名列表生成器
filename_queue = tf.train.string_input_producer(data_files, shuffle=True)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
features = tf.parse_single_example(serialized_example,
                                   features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'img_raw': tf.FixedLenFeature([], tf.string),
                                       'img_width': tf.FixedLenFeature([], tf.int64),
                                       'img_height': tf.FixedLenFeature([], tf.int64),
                                   })  # 取出包含image和label的feature对象

# tf.decode_raw可以将字符串解析成图像对应的像素数组
image = tf.decode_raw(features['img_raw'], tf.uint8)
height = tf.cast(features['img_height'], tf.int32)
width = tf.cast(features['img_width'], tf.int32)
label = tf.cast(features['label'], tf.int32)
channel = 3
image = tf.reshape(image, [height, width, channel])
img = tf.image.resize_images(image, [224, 224], method=tf.image.ResizeMethod.AREA)

batch_size = 200
mini_after_dequeue = 100
capacity = mini_after_dequeue + 3 * batch_size
# reader = tf.TFRecordReader()

example_batch, label_batch = tf.train.shuffle_batch(
    [img, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=mini_after_dequeue)


with tf.Session() as sess:  # 开始一个会话
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    threads = tf.train.start_queue_runners(sess=sess)
    # 启动多线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    x_train, y_train = sess.run([example_batch, label_batch])
    i = 0
    while not len(x_train) == 0:
    # for i in range(15):
        print x_train[i].shape
        print y_train[i]
        #image_down = np.asarray(image_down.eval(), dtype='uint8')
        print image.eval().shape
        # plt.imshow(image.eval())
        # plt.show()
        # single, l = sess.run([img, label])  # 在会话中取出image和label
        x_train, y_train = sess.run([example_batch, label_batch])
        i += 1
        print i
        # print single.shape
        # print(single.shape,l)
    coord.request_stop()
    coord.join(threads)
