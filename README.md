# NNC
Fashion mnist and Cifar10 classification using convolutional Networks

Bulding and convolutional network model and using Fahion mnist and Cifar10 objects for classification. 
Model gets tuned hyper-parameters to build the best model and plots loss/accuracy per epoch.

The file dataloder.py loades either fashion_mnist or cifar10 and fcNetwork.py builds tuner and model.
Run main.py to tune , train and plot the model. 

I tuned the model for 40 epochs and the result architecture of model is:

Best val_accuracy So Far: 0.9275833368301392
Total elapsed time: 14h 56m 45s
INFO:tensorflow:Oracle triggered exit

The hyperparameter search is complete. 


Results
=======
optimal number of output filters in the 1st convolution : 96

optimal first dropout rate                              : 0.1

optimal number of output filters in the 2nd convolution : 48

optimal second dropout rate                             : 0.30000000000000004

optimal number of units in the densely-connected layer  : 96

optimal learning rate for the optimizer                 : 0.0001

_________________________________________________________________

| Layer  | Shape | Params # |
| ------------- | ------------- | ------------- |
| 1st-convolution (Conv2D)  | (None, 28, 28, 96)  | 960 |
| 1st-max-pooling (MaxPooling2D)  | (None, 14, 14, 96)  | 0 |
| 1st-dropout (Dropout)  | (None, 14, 14, 96)  | 0 |
| 2nd-convolution (Conv2D)  | (None, 14, 14, 48)  | 41520 |
| 2nd-max-pooling (MaxPooling 2D)  | (None, 7, 7, 48)  | 0 |
| 2nd-dropout (Dropout)  | (None, 7, 7, 48)  | 0 |
| flatten-layer (Flatten)  | (None, 2352)  | 0 |
| dense-layer (Dense)  | (None, 96)  | 225888 |
| batch_normalization  | (None, 96)  | 384 |
| output-layer (Dense)  | (None, 10)  | 970 |

Total params: 269,722
Trainable params: 269,530
Non-trainable params: 192
_________________________________________________________________
I used this model to train it over 50 epochs. The result accuracies are:

Train Loss          : 0.26330

Validation Loss     : 0.32813

Test Loss           : 0.34702

Train Accuracy      : 0.94431

Validation Accuracy : 0.92542

Test Accuracy       : 0.91900

The graph of loss/accuracy per epoch is plotted:

![loss](https://user-images.githubusercontent.com/108633576/209683413-4245da77-db6b-4eb1-9111-1701a9ff6669.png)

![accuracy](https://user-images.githubusercontent.com/108633576/209683442-28f9eb2b-46ae-4d05-9d72-0e785a1288fb.png)

I have also uploaded the tuned parameters and trained weights in the link below.




