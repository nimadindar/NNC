import gc
import itertools

import numpy as np
import keras_tuner as kt
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import dataloader

SEED = 123456
np.random.seed(SEED)
tf.random.set_seed(SEED)


class fcNetwork:
    def __init__(self, epochs, reg_epoch, dataset):

        self.epochs = epochs
        self.reg_epoch = reg_epoch
        self.dataset = dataset
    

    def build_data(self, dataset):

        ''' loades datasets fashion_mnist or Cifar10 '''

        if(dataset == "fashion_mnist"):
            (train_images, train_labels), (test_images, test_labels) = dataloader.fmdataloader()
        elif(dataset == "cifar10"):
            (train_images, train_labels), (test_images, test_labels) = dataloader.cifardataloader()
        else:
            raise Exception("Data Set Name is not valid: fashion_mnist , cifar10")

        return (train_images, train_labels), (test_images, test_labels)

    def clean_up(self,mdoel_):
        tf.keras.backend.clear_session()
        del model_
        gc.collect()
    
    def build_model(self, hp):

        ''' Uses the hyper-parameters to build model '''

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=hp.Int("1st-filter", min_value=32, max_value=128, step=16),
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="same",
                    kernel_regularizer="l2",
                    dilation_rate=(1, 1),
                    activation="relu",
                    input_shape=(28, 28, 1),
                    name="1st-convolution",
                ),
                tf.keras.layers.MaxPool2D(
                    pool_size=(2, 2), strides=(2, 2), padding="same", name="1st-max-pooling"
                ),
                tf.keras.layers.Dropout(
                    rate=hp.Float("1st-dropout", min_value=0.0, max_value=0.4, step=0.1),
                    name="1st-dropout",
                ),
                tf.keras.layers.Conv2D(
                    filters=hp.Int("2nd-filter", min_value=32, max_value=64, step=16),
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="same",
                    kernel_regularizer="l2",
                    dilation_rate=(1, 1),
                    activation="relu",
                    name="2nd-convolution",
                ),
                tf.keras.layers.MaxPool2D(
                    pool_size=(2, 2), strides=(2, 2), padding="same", name="2nd-max-pooling"
                ),
                tf.keras.layers.Dropout(
                    rate=hp.Float("2nd-dropout", min_value=0.0, max_value=0.4, step=0.1),
                    name="2nd-dropout",
                ),
                tf.keras.layers.Flatten(name="flatten-layer"),
                tf.keras.layers.Dense(
                    units=hp.Int("dense-layer-units", min_value=32, max_value=128, step=16),
                    kernel_regularizer="l2",
                    activation="relu",
                    name="dense-layer",
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(units=10, activation="softmax", name="output-layer"),
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=hp.Choice(
                    "learning-rate", values=[1e-3, 1e-4, 2 * 1e-4, 4 * 1e-4]
                )
            ),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        return model

    def tuner(self, dataset,reg_epoch):


        tuner = kt.Hyperband(
            self.build_model,
            objective="val_accuracy",
            max_epochs= reg_epoch,
            seed = SEED,
            directory = "hparam-tuning",
            project_name = "%s Classification"% dataset
        )

        return tuner

    def get_hpram(self, reg_epoch, dataset):

        ''' Uses tuner to find best set of hyper-parameters in a given space '''
        
        stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
        (train_images, train_labels), (test_images, test_labels) = self.build_data(dataset)

        tuner = self.tuner(dataset, reg_epoch)
        tuner.search(
            train_images, train_labels, epochs=reg_epoch, validation_split=0.2, callbacks=[stop_early])

        best_hprams = tuner.get_best_hyperparameters(num_trials=1)[0]

        return best_hprams

    def train(self,reg_epoch ,epochs,hpram, dataset):

        (train_images, train_labels), (test_images, test_labels) = self.build_data(dataset)

        tuner = self.tuner(dataset, reg_epoch)

        model = tuner.hypermodel.build(hpram)
        history = model.fit(train_images, train_labels, epochs = epochs, validation_split = 0.2)

        return history

    def plot_history(self, hs, epochs, metric):

        ''' Plots Accuracy/loss per epoch '''

        print()
        plt.style.use("dark_background")
        plt.rcParams["figure.figsize"] = [15, 8]
        plt.rcParams["font.size"] = 16
        plt.clf()
        for label in hs:
            plt.plot(
                hs[label].history[metric],
                label="{0:s} train {1:s}".format(label, metric),
                linewidth=2,
            )
            plt.plot(
                hs[label].history["val_{0:s}".format(metric)],
                label="{0:s} validation {1:s}".format(label, metric),
                linewidth=2,
            )
        x_ticks = np.arange(0, epochs + 1, epochs / 10)
        x_ticks[0] += 1
        plt.xticks(x_ticks)
        plt.ylim((0, 1))
        plt.xlabel("Epochs")
        plt.ylabel("Loss" if metric == "loss" else "Accuracy")
        plt.legend()
        plt.show()

