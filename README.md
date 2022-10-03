### Project 1 : Warming Up with MNIST
This project aims to detect and recognize MNIST digit.\
I randomly pasted each 28x28 digit in a 75x75 black background and trained a Darknet-like model to reproduce a YoloV1 algorithm on MNIST. The model handle a unique bounding box and tries to localize and classify a digit with a 6x6 grid.

The model is composed of 5 CNN blocks with batch normalizations and LeakyReLU. It has been trained on 10 epochs, on the 60k MNIST training set and test on the 10k MNIST validation set. 

The mAP has not been produced since it's a first approach on YoloV1 but for your information, the model achieved a 97.23% validation class accracy and a validation MSE of 0.14 on confidence score (confidence of detecting an object within a bounding box).

![alt text](https://github.com/ThOpaque/Food_Recognition/blob/main/WarmingUp_with_MNIST/MNIST_localization_10exemples.png)

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
YoloMNIST                                [64, 6, 6, 5]             --
├─Sequential: 1-1                        [64, 128, 6, 6]           --
│    └─CNNBlock: 2-1                     [64, 32, 37, 37]          1,632
│    └─MaxPool2d: 2-2                    [64, 32, 18, 18]          --
│    └─CNNBlock: 2-3                     [64, 128, 16, 16]         37,120
│    └─MaxPool2d: 2-4                    [64, 128, 8, 8]           --
│    └─CNNBlock: 2-5                     [64, 64, 8, 8]            8,320
│    └─CNNBlock: 2-6                     [64, 128, 6, 6]           73,984
│    └─CNNBlock: 2-7                     [64, 128, 6, 6]           147,710
├─Sequential: 1-2                        [64, 540]                 --
│    └─Linear: 2-9                       [64, 4096]                18,878,464
│    └─LeakyReLU: 2-10                   [64, 4096]                --
│    └─Linear: 2-11                      [64, 540]                 2,212,380
==========================================================================================
Trainable params: 21,359,612
==========================================================================================
```


### Project 2 : Food Recognition on companie meal trays
(In process...)
