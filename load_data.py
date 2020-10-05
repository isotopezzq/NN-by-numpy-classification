import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def load_data():
      mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
      train_nums = mnist.train.num_examples
      validation_nums = mnist.validation.num_examples
      test_nums = mnist.test.num_examples
      print('MNIST数据集的个数')
      print(' >>>train_nums=%d' % train_nums,'\n',
            '>>>validation_nums=%d'% validation_nums,'\n',
            '>>>test_nums=%d' % test_nums,'\n')

      train_data = mnist.train.images   #所有训练数据
      val_data = mnist.validation.images  #(5000,784)
      test_data = mnist.test.images       #(10000,784)
      #print('>>>训练集数据大小：',train_data.shape,'\n',
            #'>>>一副图像的大小：',train_data[0].shape)

      train_labels = mnist.train.labels     #(55000,10)
      val_labels = mnist.validation.labels  #(5000,10)
      test_labels = mnist.test.labels       #(10000,10)
      return mnist
      '''
      print('>>>训练集标签数组大小：',train_labels.shape,'\n',
      '>>>一副图像的标签大小：',train_labels[1].shape,'\n',
      '>>>一副图像的标签值：',train_labels[0])
      batch_size = 500
      batch_xs,batch_ys = mnist.train.next_batch(batch_size)
      print('使用mnist.train.next_batch(batch_size)批量读取样本\n')
      print('>>>批量读取100个样本:数据集大小=',batch_xs.shape,'\n',
            '>>>批量读取100个样本:标签集大小=',batch_ys.shape)
      '''

'''
mnist = load_data()
train_data = mnist.train.images
print(train_data[0])
'''