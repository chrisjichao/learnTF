import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image


class MNISTLoader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label),(self.test_data,self.test_label) = mnist.load_data()
        self.train_data = np.expand_dims(self.train_data.astype(np.float32)/ 255., axis=-1)
        self.test_data = np.expand_dims(self.test_data.astype(np.float32)/ 255., axis=-1)
        self.train_label = self.train_label.astype(np.float32)
        self.test_label = self.test_label.astype(np.float32)
        self.num_train_data,self.num_test_data = self.train_data.shape[0],self.test_data.shape[0]
        print(np.shape(self.train_data))

    def get_batch(self,batch_size):
        index = np.random.randint(0,self.num_train_data,batch_size)
        return  self.train_data[index,:],self.train_label[index]




class MLP(tf.keras.Model):
    def __int__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=100,activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs, training=None, mask=None):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(2)
        return tf.nn.softmax(x)

loader_data = MNISTLoader()
loader_data.get_batch(10)