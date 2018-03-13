import os
import Queue
import shutil
import argparse
import sys

# ori_path = '/media/_Data/OrigPic'
#
# train_path = '/media/_Data/multil_classification'
#
# sex_path = '/media/_Data/OrigPic/sexy-ok-porn/train_60w/bad'
#
# train_path = os.path.join(train_path, 'original')
# sex_train_path = os.path.join(train_path, 'sex')
# test_path = os.path.join(train_path, 'test')
# sex_test_path = os.path.join(train_path, 'sex')
sex_q = Queue.Queue()


def init_floder(paths_list):
    for i in range(9000):
        path = paths_list[i]
        if not os.path.exists(path):
            os.mkdir(path)


def set_sex(path):
    path_exp = os.path.expanduser(path)
    classes = os.listdir(path_exp)
    classes.sort()
    for i in range(10000):
        image_path = os.path.join(path_exp, classes[i])
        sex_q.put(image_path)
        print('puting {} image'.format(image_path))


def copy_sex(sex_train, sex_test):
    j = 0
    while not sex_q.empty():
        if j < 9001:
            img_path = sex_q.get()
            basename = os.path.basename(img_path)
            dst_path = os.path.join(sex_train, basename)
            shutil.copy(img_path, dst_path)
            print('copy {} image'.format(dst_path))
            j += 1
        else:
            img_path = sex_q.get()
            basename = os.path.basename(img_path)
            dst_path = os.path.join(sex_test, basename)
            shutil.copy(img_path, dst_path)
            print('copy {} image'.format(dst_path))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str,
                        help='Directory where to write event logs.', default='~/logs/facenet')
    parser.add_argument('--train_path', type=str,
                        help='Directory where to write trained models and checkpoints.', default='~/models/facenet')
    parser.add_argument('--test_path', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)


if __name__ == '__main__':
    arg = parse_arguments(sys.argv[1:])
    path = arg.path
    train_path = arg.train_path
    test_path = arg.test_path
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)
    set_sex(path)
    copy_sex(train_path, test_path)
