import gc
import itertools

import numpy as np
import keras_tuner as kt
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import dataloader

from fcNetwork import fcNetwork as fcn

train = True
tune = True
plot = True
reg_epoch = 2
train_epoch = 10
dataset = "cifar10"

fcn = fcn(train_epoch, reg_epoch, dataset)


if(tune):

    # Tuning hyper parameters of model and chosing best parameters.
    
    best_hprams = fcn.get_hpram(reg_epoch, dataset)

    print(
            f"""
        The hyperparameter search is complete. \n

        Results
        =======
        |
        ---- optimal number of output filters in the 1st convolution : {best_hprams.get('1st-filter')}
        |
        ---- optimal first dropout rate                              : {best_hprams.get('1st-dropout')}
        |
        ---- optimal number of output filters in the 2nd convolution : {best_hprams.get('2nd-filter')}
        |
        ---- optimal second dropout rate                             : {best_hprams.get('2nd-dropout')}
        |
        ---- optimal number of units in the densely-connected layer  : {best_hprams.get('dense-layer-units')}
        |
        ---- optimal learning rate for the optimizer                 : {best_hprams.get('learning-rate')}
        """
    )

if(train):

    # Uses the best hyper parameters to train the model.

    # (train_images, train_labels), (test_images, test_labels) = fcn.build_data(dataset[0])

    history = fcn.train(reg_epoch, train_epoch, best_hprams, dataset)

    val_acc_per_epoch = history.history["val_accuracy"]
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print("Best epoch: %d" % (best_epoch,))
    # eval_result = model.evaluate(test_images, test_labels, verbose=3)
    # print("[test loss, test accuracy]:", eval_result)


if(plot):

    # Plots Loss and Accuracy per epoch. 

    print("Train Loss          : {0:.5f}".format(history.history["loss"][-1]))
    print("Validation Loss     : {0:.5f}".format(history.history["val_loss"][-1]))
    # print("Test Loss           : {0:.5f}".format(eval_result[0]))
    print("-------------------")
    print("Train Accuracy      : {0:.5f}".format(history.history["accuracy"][-1]))
    print("Validation Accuracy : {0:.5f}".format(history.history["val_accuracy"][-1]))
    # print("Test Accuracy       : {0:.5f}".format(eval_result[1]))

    # Plot train and validation error per epoch.
    fcn.plot_history(hs={"CNN": history}, epochs=best_epoch, metric="loss")
    fcn.plot_history(hs={"CNN": history}, epochs=best_epoch, metric="accuracy")