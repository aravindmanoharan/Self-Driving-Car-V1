# Self-Driving-Car-V1

This is an implementation of a CNN architecture which predicts steering angles for a self driving car. This project is inspired from the [Udacity Self Driving Car](https://github.com/udacity/CarND-Behavioral-Cloning-P3) and [End to End Learning for Self-Driving Cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) by NVIDIA.

## Dataset

The Dataset can be downloaded from [here](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)

## Requirements

This CNN model is implemented using TensorFlow framework.

```
pip install tensorflow
```

Please check the requirements.txt file for all the packages which you need to install. Most of the packages can be installed using the pip package manager.

## CNN Architecture

The architecture is inspired by this [NVIDIA paper](https://arxiv.org/abs/1604.07316) on End to End Learning for Self-Driving Cars. But since our dataset is not very much complex, we just used a simple architecture than what was implemented in that paper. The architecture consists of three convolutional layers with a kernel size of 3 x 3 with no strides and followed by three fully connected layers. The convolutional layers may perform as feature extractions and the the fully connected layers may function as a controller for steering.

## Data Preprocessing

This dataset has three images (Left, Center and Right) for every timestamp with the same steering angle. But the model inputs only one image for each steering angle. Hence for the left and right images we alter the steering angle by an alpha value. This helps us to increasing the dataset by three times. After this, we perform data augmentation by mirroring each image and it's steering angles. This again increases the dataset by two times.

## Training the model

The model achieves good results by just training the model for 5 epoch. We use mean squared error as our cost function. The model is trained on 80% of the data and the mean square error for the test data is 0.8%. Let's see how it performs on our dataset (The output shown below is an edited GIF with more number of frames per second).

![](demo.gif)

## Reference

1. Bojarski, Mariusz, Davide Del Testa, Daniel Dworakowski, Bernhard Firner, Beat Flepp, Prasoon Goyal, Lawrence D. Jackel et al. "End to end learning for self-driving cars." arXiv preprint arXiv:1604.07316 (2016).

## Have questions? Need help with the code?

If you're having issues with or have questions about the code, [file an issue](https://github.com/aravindmanoharan/Self-Driving-Car-V1/issues) in this repository so that I can get back to you soon.
