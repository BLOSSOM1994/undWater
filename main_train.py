
from model_train import T_CNN
from utils import (
  imsave,
  prepare_data
)
import numpy as np
import tensorflow as tf

import os





def main(_):
  #pp.pprint(__flags)
  checkpoint_dir='/content/drive/MyDrive/DataSet_WaterNet/checkpoint'
  sample_dir='/content/drive/MyDrive/DataSet_WaterNet/challenging-60'
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
  
  with tf.Session() as sess:
    srcnn = T_CNN(sess, 
                  image_height=112,
                  image_width=112, 
                  label_height=112, 
                  label_width=112, 
                  batch_size=16,
                  c_dim=3, 
                  checkpoint_dir=checkpoint_dir,
                  sample_dir=sample_dir
                  )

    srcnn.train()
    
if __name__ == '__main__':
  tf.app.run()
