## Chongyi Li, Chunle Guo, Wenqi Ren, Runmin Cong, Junhui Hou, Sam Kwong, Dacheng Tao , "An Underwater Image Enhancement Benchmark Dataset and Beyond" IEEE TIP 2019 #######
## Project: https://li-chongyi.github.io/proj_benchmark.html 
############################################################################################################################################################################

from model import T_CNN
from utils import *
import numpy as np
import tensorflow as tf

import pprint
import os
import pandas as pd


def main(_):
  checkpoint_dir='/content/drive/MyDrive/DataSet_WaterNet/checkpoint'
  sample_dir='/content/drive/MyDrive/DataSet_WaterNet/challenging-60'
  FLAGS=pd.DataFrame()
  FLAGS["epoch"]=[4]
  FLAGS["batch_size"]=[128]
  FLAGS["image_height"] = [ 112]
  FLAGS["image_width"] = [ 112]
  FLAGS["label_height"] = [ 112]
  FLAGS["label_width"] = [ 112]
  FLAGS["learning_rate"] = [ 0.001]
  FLAGS["beta1"] = [ 0.5]
  FLAGS["c_dim"] = [ 3]
  FLAGS["checkpoint_dir"] = [ '/content/drive/MyDrive/DataSet_WaterNet/checkpoint']
  FLAGS["sample_dir"] = [ '/content/drive/MyDrive/DataSet_WaterNet/challenging-60']
  FLAGS["test_data_dir"] = [ "/content/drive/MyDrive/DataSet_WaterNet/output"]
  FLAGS["is_train"] = [ False]





  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)
  filenames = os.listdir('/content/drive/MyDrive/DataSet_WaterNet/test_real')
  data_dir = os.path.join(os.getcwd(), '/content/drive/MyDrive/DataSet_WaterNet/test_real')
  data = glob.glob(os.path.join(data_dir, "*.png"))
  test_data_list = data + glob.glob(os.path.join(data_dir, "*.jpg"))+glob.glob(os.path.join(data_dir, "*.bmp"))+glob.glob(os.path.join(data_dir, "*.jpeg"))

  filenames1 = os.listdir('/content/drive/MyDrive/DataSet_WaterNet/wb_real')
  data_dir1 = os.path.join(os.getcwd(), '/content/drive/MyDrive/DataSet_WaterNet/wb_real')
  data1 = glob.glob(os.path.join(data_dir1, "*.png"))
  test_data_list1 = data1 + glob.glob(os.path.join(data_dir1, "*.jpg"))+glob.glob(os.path.join(data_dir1, "*.bmp"))+glob.glob(os.path.join(data_dir1, "*.jpeg"))

  filenames2 = os.listdir('/content/drive/MyDrive/DataSet_WaterNet/ce_real')
  data_dir2 = os.path.join(os.getcwd(), '/content/drive/MyDrive/DataSet_WaterNet/ce_real')
  data2 = glob.glob(os.path.join(data_dir2, "*.png"))
  test_data_list2 = data2 + glob.glob(os.path.join(data_dir2, "*.jpg"))+glob.glob(os.path.join(data_dir2, "*.bmp"))+glob.glob(os.path.join(data_dir2, "*.jpeg"))

  filenames3 = os.listdir('/content/drive/MyDrive/DataSet_WaterNet/gc_real')
  data_dir3 = os.path.join(os.getcwd(), '/content/drive/MyDrive/DataSet_WaterNet/gc_real')
  data3 = glob.glob(os.path.join(data_dir3, "*.png"))
  test_data_list3 = data3 + glob.glob(os.path.join(data_dir3, "*.jpg"))+glob.glob(os.path.join(data_dir3, "*.bmp"))+glob.glob(os.path.join(data_dir3, "*.jpeg"))

  for ide in range(0,len(test_data_list)):
    image_test =  get_image(test_data_list[ide],is_grayscale=False)
    wb_test =  get_image(test_data_list1[ide],is_grayscale=False)
    ce_test =  get_image(test_data_list2[ide],is_grayscale=False)
    gc_test =  get_image(test_data_list3[ide],is_grayscale=False)
    shape = image_test.shape
    tf.reset_default_graph()
    with tf.Session() as sess:
      # with tf.device('/cpu:0'):
        srcnn = T_CNN(sess, 
                  image_height=shape[0],
                  image_width=shape[1],  
                  label_height=FLAGS.label_height, 
                  label_width=FLAGS.label_width, 
                  batch_size=FLAGS.batch_size,
                  c_dim=FLAGS.c_dim, 
                  c_depth_dim=FLAGS.c_depth_dim,
                  checkpoint_dir=FLAGS.checkpoint_dir,
                  sample_dir=FLAGS.sample_dir,
                  test_image_name = test_data_list[ide],
                  test_wb_name = test_data_list1[ide],
                  test_ce_name = test_data_list2[ide],
                  test_gc_name = test_data_list3[ide],
                  id = ide
                  )

        srcnn.train(FLAGS)
        sess.close()
    tf.get_default_graph().finalize()
if __name__ == '__main__':
  tf.app.run()
