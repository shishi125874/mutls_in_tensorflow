import cv2
import os
import imghdr


def resize(img_list, output):
    for img in img_list:
        image = cv2.imread(img)
        res = cv2.resize(image, (160, 160), interpolation=cv2.INTER_CUBIC)
        dst_path = os.path.join(output, os.path.basename(img))
        cv2.imwrite(dst_path, res)


def is_img(list_a):
    list_b = list_a.copy()
    for i in range(len(list_a)):
        image_type = imghdr.what(list_a[i])
        if image_type == None or image_type == 'gif':
            list_b.remove(list_a[i])
        else:
            continue
    return list_b


def combine_path(path, name):
    dst_name = os.path.join(path, name)
    if not os.path.exists(dst_name):
        os.mkdir(dst_name)
    return dst_name


def run(path, output):
    dirs = os.listdir(path)
    for item in dirs:
        forlder = os.path.join(path, item)
        images = os.listdir(forlder)
        output_forlder = combine_path(output, item)

        rust_list = []
        for img in images:
            image = os.path.join(forlder, img)
            rust_list.append(image)
        result_list = is_img(rust_list)
        resize(result_list, output_forlder)


if __name__ == '__main__':
    path = '/media/_Data/multil_classification/original'
    output = '/media/_Data/multil_classification/deleted'
    run(path, output)

    path = '/media/_Data/multil_classification/test'
    output = '/media/_Data/multil_classification/deleted-test'
    run(path, output)
