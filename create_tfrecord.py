# encoding: utf-8
import time
import os
from PIL import Image
import tensorflow as tf
import time

#图片路径
cwd = '/media/_Data/guoxc/trainData'
#文件路径
filepath = '/media/_Data/multil_classification/tfrecord'
#存放图片个数
bestnum = 10000
#第几个图片
num = 0
#第几个TFRecord文件
recordfilenum = 0
#类别
classes=['5_txt','0_bad','1_good','2_baoluan','3_guns','4_peoples']

#tfrecords格式文件名
ftrecordfilename = ("traindata.tfrecords-%.3d" % recordfilenum)
writer= tf.python_io.TFRecordWriter(filepath+'/'+ftrecordfilename)
#类别和路径
for index,name in enumerate(classes):
    print(index)
    print(name)
    class_path = os.path.join(cwd,name)
    for img_name in os.listdir(class_path):
        num=num+1
        if num>bestnum:
            print('over 1 tfrecord')
            num = 1
            recordfilenum = recordfilenum + 1
            #tfrecords格式文件名
            ftrecordfilename = ("traindata.tfrecords-%.3d" % recordfilenum)
            writer= tf.python_io.TFRecordWriter(filepath+'/'+ftrecordfilename)
        #print('路径',class_path)
        #print('第几个图片：',num)
        #print('文件的个数',recordfilenum)
        #print('图片名：',img_name)

        img_path = os.path.join(class_path,img_name)
        img=Image.open(img_path,'r')
        size = img.size
 #        print(size[1],size[0])
  #       print(size)
        #print(img.mode)
        img_raw=img.tobytes()#将图片转化为二进制格式
        example = tf.train.Example(
             features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            'img_width':tf.train.Feature(int64_list=tf.train.Int64List(value=[size[0]])),
            'img_height':tf.train.Feature(int64_list=tf.train.Int64List(value=[size[1]]))
        }))
        writer.write(example.SerializeToString())  #序列化为字符串
writer.close()

