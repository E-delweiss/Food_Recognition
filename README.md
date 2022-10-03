### Project 1 : Warming Up with MNIST
This project aims to detect and recognize MNIST digit.\
I randomly pasted each 28x28 digit in a 75x75 black background and trained a Darknet-like model to reproduce a YoloV1 algorithm on MNIST. The model handle a unique bounding box and tries to localize and classify a digit with a 6x6 grid.

The model is composed of [N] parameters with [N] layers. It has been trained on 10 epochs, on the 60k MNIST training set and test on the 10k MNIST validation set. 

The mAP has not been produced since it's a first approach on YoloV1 but for your information, the model achieved a 97.23% validation class accracy and a validation MSE of 0.14 on confidence score (confidence of detecting an object within a bounding box).

![alt text](https://github.com/ThOpaque/Food_Recognition/blob/main/WarmingUp_with_MNIST/MNIST_localization_10exemples.png)


### Project 2 : Food Recognition on companie meal trays
(In process...)
