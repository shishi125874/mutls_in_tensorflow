import os
import shutil


def run(path,output):
    if not os.path.exists(output):
        os.mkdir(output)
    classes = os.listdir(path)
    for i in range(1000):
        image = os.path.join(path, classes[i])
        shutil.move(image, output)

if __name__ == '__main__':
    path = '/media/_Data/multil_classification/original/peoples'
    dst_path = '/media/_Data/multil_classification/test/peoples'
    run(path,dst_path)