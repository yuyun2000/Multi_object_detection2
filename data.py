from makelabel import fusion_4img
import numpy as np
import tensorflow as tf
import os
import cv2

def load_mnist():
    img_list = os.listdir('./train/img')
    label_list= os.listdir('./train/label')
    img_list = sorted(img_list)
    label_list = sorted(label_list)
    return img_list,label_list

def get_img_label(img,xml):

    image = cv2.imread('./train/img/%s'%img.numpy().decode()).astype(np.float32)
    image = image / 255
    label = np.load('./train/label/%s.npz.npy'%img.numpy().decode()[:-4])
    # print(label)
    return image,label

def train_iterator():
    img,label = load_mnist()
    dataset = tf.data.Dataset.from_tensor_slices((img,label)).shuffle(len(img))
    dataset = dataset.repeat()
    dataset = dataset.map(lambda x, y: tf.py_function(get_img_label,inp=[x,y],Tout=[tf.float32,tf.float32]),num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(128)


    it = dataset.__iter__()
    return it

if __name__ == '__main__':

    it = train_iterator()
    images, labels = it.next()
    print(labels[0])

    # print(tf.reshape(labels[0:1,:,:,0:1],(8,8)))


