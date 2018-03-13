import os
import shutil

path = '/media/_Data/multil_classification/original/baokong'
dst_path = '/media/_Data/multil_classification/original/kongbu'
test_path = '/media/_Data/multil_classification/test/kongbu'

if not os.path.exists(dst_path):
    os.mkdir(dst_path)

if not os.path.exists(test_path):
    os.mkdir(test_path)

path_exp = os.path.expanduser(path)

list_class = os.listdir(path_exp)

for i in range(10000):
    image = os.path.join(path_exp, list_class[i])
    if i < 9000:
        shutil.copy(image, dst_path)
    else:
        shutil.copy(image, test_path)
