import tensorflow as tf
import tensorflow_datasets as tfds

def fmdataloader():

    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    return (train_images, train_labels), (test_images, test_labels)


def cifardataloader():
    
    cifar10 = tf.keras.datasets.cifar10
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    return (train_images, train_labels), (test_images, test_labels)
