# coding=utf-8

# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys
import random
import tensorflow as tf
import numpy as np
import importlib
import argparse
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import deal_data
import util
import cv2
import math


def main(args):
    network = importlib.import_module(args.model_def)

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    # 设置log日志
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # Store some git revision info in a text file in the log directory
    src_path, _ = os.path.split(os.path.realpath(__file__))
    # facenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)
    random.seed(args.seed)
    # 获取数据集,通过get_dataset获取的train_set是包含文件路径与标签的集合
    train_set = deal_data.get_dataset(args.data_dir)
    test_set = deal_data.get_dataset(args.test_dir)
    nrof_classes = len(train_set)

    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)

        # Get a list of image paths and their labels
        # 获取图片地址和标签
        image_list, label_list = deal_data.get_image_paths_and_labels(train_set)
        test_image_list, test_label_list = deal_data.get_image_paths_and_labels(test_set)
        assert len(image_list) > 0, 'The dataset should not be empty'

        # Create a queue that produces indices into the image_list and label_list
        labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
        range_size = array_ops.shape(labels)[0]
        index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
                                                    shuffle=True, seed=None, capacity=32)

        index_dequeue_op = index_queue.dequeue_many(args.batch_size * args.epoch_size, 'index_dequeue')

        # 学习率
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

        # 批大小
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')

        # 用于判断是训练还是测试
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

        # 图像路径
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='image_paths')

        # 图像标签
        labels_placeholder = tf.placeholder(tf.int64, shape=(None, 1), name='labels')
        # labels_placeholder = tf.reshape(labels_placeholder, shape=(-1, 1))

        # 新建一个队列,数据流操作,fifo,先入先出
        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                              dtypes=[tf.string, tf.int64],
                                              shapes=[(1,), (1,)],
                                              shared_name=None, name=None)
        # enqueue_many返回的是一个操作
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder], name='enqueue_op')

        nrof_preprocess_threads = 4
        images_and_labels = []
        for _ in range(nrof_preprocess_threads):
            filenames, label = input_queue.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                # 读图像
                image = tf.image.decode_image(file_contents, channels=3)
                if args.random_rotate:
                    image = tf.py_func(random_rotate_image, [image], tf.uint8)
                if args.random_crop:
                    image = tf.random_crop(image, [args.image_size, args.image_size, 3])
                else:
                    image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
                if args.random_flip:
                    image = tf.image.random_flip_left_right(image)

                # pylint: disable=no-member
                image.set_shape((args.image_size, args.image_size, 3))
                images.append(tf.image.per_image_standardization(image))
            # 读取后的图片和标签list
            images_and_labels.append([images, label])

        image_batch, label_batch = tf.train.batch_join(
            images_and_labels, batch_size=batch_size_placeholder,
            shapes=[(args.image_size, args.image_size, 3), ()], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * args.batch_size,
            allow_smaller_final_batch=True)
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        label_batch = tf.identity(label_batch, 'label_batch')

        print('Total number of classes: %d' % nrof_classes)
        print('Total number of examples: %d' % len(image_list))

        print('Building training graph')

        # Build the inference graph
        # 创建网络图:除了全连接层和损失层
        prelogits, _ = network.inference(image_batch, args.keep_probability,
                                         phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size,
                                         weight_decay=args.weight_decay)
        # 全联接
        logits = slim.fully_connected(prelogits, 4, activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      weights_regularizer=slim.l2_regularizer(args.weight_decay),
                                      scope='Logits', reuse=False)
        predicts_soft = tf.nn.softmax(logits=logits, dim=-1)

        # 对维度dim进行L2范式标准化 output = x / sqrt(max(sum(x**2), epsilon))
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # train_prediction = tf.nn.softmax(prelogits)

        # Add center loss
        # if args.center_loss_factor > 0.0:
        #     prelogits_center_loss, _ = util.center_loss(prelogits, label_batch, args.center_loss_alfa, nrof_classes)
        #     tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * args.center_loss_factor)

        # 将指数衰减应用到学习率上
        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                                                   args.learning_rate_decay_epochs * args.epoch_size,
                                                   args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Calculate the average cross entropy loss across the batch
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_batch, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        # Calculate the total losses
        # 根据REGULARIZATION_LOSSES返回一个收集器中所收集的值的列表
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # 我们选择L2-正则化来实现这一点，L2正则化将网络中所有权重的平方和加到损失函数。如果模型使用大权重，则对应重罚分，并且如果模型使用小权重，则小罚分。
        # 这就是为什么我们在定义权重时使用了regularizer参数，并为它分配了一个l2_regularizer。这告诉了TensorFlow要跟踪
        # l2_regularizer这个变量的L2正则化项（并通过参数reg_constant对它们进行加权）。
        # 所有正则化项被添加到一个损失函数可以访问的集合——tf.GraphKeys.REGULARIZATION_LOSSES。
        # 将所有正则化损失的总和与先前计算的triplet_loss相加，以得到我们的模型的总损失。
        total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        # 确定优化方法并求根据损失函数求梯度,在这里面,每更新一次参数,global_step会加1
        train_op = util.train(total_loss, global_step, args.optimizer,
                              learning_rate, args.moving_average_decay, tf.global_variables(), args.log_histograms)

        # top_k_op = tf.nn.in_top_k(predicts_soft, labels_placeholder, 1)
        correct_predict = tf.equal(tf.argmax(predicts_soft, 1), tf.argmax(labels_placeholder, 1))
        # Create a saver
        # 创建一个saver用于保存或从内存中恢复一个模型参数
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        # 能够在gpu上分配的最大内存
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        # 获取线程坐标
        coord = tf.train.Coordinator()
        # 将队列中的所有sunner开始执行
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():

            # Training and validation loop
            print('Running training')
            epoch = 0
            # 将所有数据过一遍的次数
            while epoch < args.max_nrof_epochs:
                # 这里是返回当前的global_step值吗,step可以看做是全局的批处理个数
                step = sess.run(global_step, feed_dict=None)
                # epoch_size是一个epoch中批的个数
                # 这个epoch是全局的批处理个数除以一个epoch中批的个数得到epoch,这个epoch将用于求学习率
                epoch = step // args.epoch_size
                # Train for one epoch
                train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder,
                      labels_placeholder,
                      learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step,
                      total_loss, train_op, summary_op, summary_writer, regularization_losses
                      )

                # Save variables and the metagraph if it doesn't exist already
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)

                if args.lfw_dir:
                    evaluate(sess, correct_predict, image_paths_placeholder, labels_placeholder, phase_train_placeholder,
                             batch_size_placeholder,
                             logits, label_batch, test_image_list, args.test_batch_size,
                             log_dir, step, summary_writer)
    return model_dir


#  image_list图片地址list，image_paths_placeholder图片地址占位符，
def train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder,
          labels_placeholder,
          learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step,
          loss, train_op, summary_op, summary_writer, regularization_losses):
    batch_number = 0

    # if args.learning_rate > 0.0:
    lr = args.learning_rate
    # else:
    #     lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)

    index_epoch = sess.run(index_dequeue_op)
    label_epoch = np.array(label_list)[index_epoch]
    image_epoch = np.array(image_list)[index_epoch]

    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_epoch), 1)
    image_paths_array = np.expand_dims(np.array(image_epoch), 1)
    # 把图片地址输入到模型
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})

    # Training loop
    train_time = 0
    # 下面的一个while循环运行一个批处理
    while batch_number < args.epoch_size:
        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder: True,
                     batch_size_placeholder: args.batch_size}
        if (batch_number % 100 == 0):
            err, _, step, reg_loss, summary_str = sess.run(
                [loss, train_op, global_step, regularization_losses, summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step)
        else:
            err, _, step, reg_loss = sess.run([loss, train_op, global_step, regularization_losses], feed_dict=feed_dict)
        duration = time.time() - start_time
        print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tRegLoss %2.3f' %
              (epoch, batch_number + 1, args.epoch_size, duration, err, np.sum(reg_loss)))
        batch_number += 1
        train_time += duration
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, step)
    return step


def random_rotate_image(image):
    angle = np.random.uniform(low=-10.0, high=10.0)
    M = cv2.getRotationMatrix2D((image.shape[0] / 2, image.shape[1] / 2), 90, 1)
    dst = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return dst
    # return misc.imrotate(image, angle, 'bicubic')


def evaluate(sess, correct_predict, image_paths_placeholder, labels_placeholder, phase_train_placeholder,
             batch_size_placeholder,
             logits, labels, image_paths, batch_size, log_dir, step, summary_writer):
    # start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Runnning forward pass on LFW images')

    true_count = 0
    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.arange(0, len(image_paths)), 1)
    image_paths_array = np.expand_dims(np.array(labels), 1)
    nrof_images = len(labels_array)
    nrof_batches = nrof_images // batch_size
    emb_array = np.zeros((nrof_images,))
    lab_array = np.zeros((nrof_images,))
    # for _ in range(nrof_batches):
    feed_dict = {phase_train_placeholder: False, batch_size_placeholder: batch_size}
    # emb, lab = sess.run([logits, labels], feed_dict=feed_dict)
    prediction = sess.run([correct_predict], feed_dict=feed_dict)
    accuracy = tf.reduce_mean(tf.cast(prediction, "float"))
        # prediction = sess.run(top_k_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
        # true_count = np.sum(prediction)

    # precidion = true_count / len(labels_array)
    print('precidion @ 1= %.3f' % accuracy)

    # with tf.name_scope('accuracy'):
    #     accuracy = tf.reduce_mean(tf.cast(prediction, "float"))
    # tf.summary.scalar('accuracy', accuracy)

    # train_accuracy = accuracy.eval(top_k_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})


def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)


def get_test_paths(dir):
    path_list = []


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--logs_base_dir', type=str,
                        help='Directory where to write event logs.', default='~/logs/facenet')
    parser.add_argument('--models_base_dir', type=str,
                        help='Directory where to write trained models and checkpoints.', default='~/models/facenet')
    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory containing aligned face patches. Multiple directories are separated with colon.',
                        default='~/datasets/casia/casia_maxpy_mtcnnalign_182_160')
    parser.add_argument('--model_def', type=str,
                        help='Model definition. Points to a module containing the definition of the inference graph.',
                        default='models.inception_resnet_v1')
    parser.add_argument('--max_nrof_epochs', type=int,
                        help='Number of epochs to run.', default=500)
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--epoch_size', type=int,
                        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--random_crop',
                        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
                             'If the size of the images in the data directory is equal to image_size no cropping is performed',
                        action='store_true')
    parser.add_argument('--random_flip',
                        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--random_rotate',
                        help='Performs random rotations of training images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
                        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
                        help='L2 weight regularization.', default=0.0)
    # parser.add_argument('--center_loss_factor', type=float,
    #                     help='Center loss factor.', default=0.0)
    # parser.add_argument('--center_loss_alfa', type=float,
    #                     help='Center update rate for center loss.', default=0.95)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate', type=float,
                        help='Initial learning rate. If set to a negative value a learning rate ' +
                             'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
                        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
                        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--nrof_preprocess_threads', type=int,
                        help='Number of preprocessing (data loading and augmentation) threads.', default=4)
    parser.add_argument('--log_histograms',
                        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--filter_filename', type=str,
                        help='File containing image data used for dataset filtering', default='')
    parser.add_argument('--filter_percentile', type=float,
                        help='Keep only the percentile images closed to its class center', default=100.0)
    parser.add_argument('--filter_min_nrof_images_per_class', type=int,
                        help='Keep only the classes with this number of examples or more', default=0)

    # Parameters for validation on LFW
    # parser.add_argument('--test_pairs', type=str,
    #                     help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    # parser.add_argument('--test_file_ext', type=str,
    #                     help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--test_dir', type=str,
                        help='Path to the data directory containing aligned face patches.', default='')
    parser.add_argument('--test_batch_size', type=int,
                        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('--test_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
