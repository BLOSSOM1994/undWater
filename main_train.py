
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
    import pandas as pd
    a=pd.DataFrame()
    a["epoch"]=[400]
    a["batch_size"]=[16]
    a["image_height"] = [ 112]
    a["image_width"] = [ 112]
    a["label_height"] = [ 112]
    a["label_width"] = [ 112]
    a["learning_rate"] = [ 0.001]
    a["beta1"] = [ 0.5]
    a["c_dim"] = [ 3]
    a["checkpoint_dir"] = [ '/content/drive/MyDrive/DataSet_WaterNet/checkpoint']
    a["sample_dir"] = [ '/content/drive/MyDrive/DataSet_WaterNet/challenging-60']
    a["test_data_dir"] = [ "/content/drive/MyDrive/waternetDataSets/output"]
    a["is_train"] = [ True]

    srcnn.train(a)
    
if __name__ == '__main__':
  tf.app.run()
