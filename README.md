![alt text](https://github.com/ThOpaque/Food_Recognition/blob/main/WarmingUp_with_MNIST/results/MNIST_localization_10exemples.png)


### Project 1 : Warming Up with MNIST
This project aims to detect and recognize MNIST digit. See the [RoadMap](https://github.com/ThOpaque/Food_Recognition/blob/main/WarmingUp_with_MNIST/RoadMap.md) notebook if needed.\
I randomly pasted each 28x28 digit in a 75x75 black background and trained a Darknet-like model to reproduce a YoloV1 algorithm on MNIST. The model handle a unique bounding box and tries to localize and classify a digit with a 6x6 grid.

The model is composed of 5 CNN blocks with batch normalizations and LeakyReLU. It has been trained on 10 epochs, on the 60k MNIST training set and test on the 10k MNIST validation set. 

The mAP has not been produced since it's a first approach on YoloV1 but for your information, the model achieved a 97.23% validation class accuracy and a validation MSE of 0.14 on confidence score (confidence of detecting an object within a bounding box).


```
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
```


### Project 2 : Food Recognition on Company Meal Trays
After understanding what's under the hood of YOLO, this project aims to detect and recognize various objects on meal trays from my former company restaurant. Make sure to see the [RoadMap] if needed. \
Each photo contains a meal tray with various objects on it. Each photo was taken from the same point of view four times : when the "photo kiosk" detects a meal tray, it captures 4 photos to cover the time spent by the employee to place his meal tray correctly.

Each meal tray contains various objects that I labelised with www.cvat.org (www.cvat.ai now). Since this exercice is a kind of a "proof of concept", and since the labelisation process is time-consumming, I labelised only about **480** images.

The goal is...

```
===============================================================================================
YoloResNet                                    [16, 7, 7, 18]            --
├─Sequential: 1-1                             [16, 512, 56, 56]         --
├─Sequential: 1-2                             [16, 512, 7, 7]           --
│    └─MaxPool2d: 2-7                         [16, 512, 28, 28]         --
│    └─Conv2d: 2-8                            [16, 1024, 28, 28]        4,719,616
│    └─Conv2d: 2-9                            [16, 512, 28, 28]         524,800
│    └─MaxPool2d: 2-10                        [16, 512, 14, 14]         --
│    └─Conv2d: 2-11                           [16, 1024, 14, 14]        4,719,616
│    └─Conv2d: 2-12                           [16, 512, 14, 14]         524,800
│    └─MaxPool2d: 2-13                        [16, 512, 7, 7]           --
├─Sequential: 1-3                             [16, 882]                 --
│    └─Flatten: 2-14                          [16, 25088]               --
│    └─Linear: 2-15                           [16, 4096]                102,764,544
│    └─LeakyReLU: 2-16                        [16, 4096]                --
│    └─Dropout: 2-17                          [16, 4096]                --
│    └─Linear: 2-18                           [16, 882]                 3,613,554
│    └─Sigmoid: 2-19                          [16, 882]                 --
===============================================================================================
Total params: 119,432,114
Trainable params: 116,866,930
Non-trainable params: 2,565,184
```

[in process]


**Disclaimer: all the photos in my dataset are NOT under licence or NDA. They were given to me by the restaurant adminitration after a lawyer approval.**

