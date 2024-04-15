# LeNet5_MLP_project


# MNIST Digit Classification

This project implements digit classification on the MNIST dataset using the PyTorch framework. It includes two models: the classic LeNet-5 and a custom Multi-Layer Perceptron (MLP) designed to have a similar number of parameters as LeNet-5. The goal is to compare the performance of these models in terms of accuracy and loss over multiple training epochs.

## Project Structure

- `dataset.py`: Contains the MNIST dataset handling including loading and transforming images.
- `model.py`: Defines the architectures of LeNet-5 and CustomMLP.
- `main.py`: Main script to execute training and testing of the models.
- `README.md`: This file, explaining the project and how to run it.



# Model Implementations and Parameter Calculations
## LeNet-5 Architecture:

Input: 1x28x28 image (since MNIST images are 28x28 pixels and grayscale)

C1: Convolutional layer with 6 filters of size 5x5, stride 1 (output: 6x24x24)

S2: Max pooling layer with size 2x2, stride 2 (output: 6x12x12)

C3: Convolutional layer with 16 filters of size 5x5, stride 1 (output: 16x8x8)

S4: Max pooling layer with size 2x2, stride 2 (output: 16x4x4)

C5: Fully connected layer with 120 units

F6: Fully connected layer with 84 units

Output: Fully connected layer with 10 units (10 classes)

### Parameter Calculation for LeNet-5:
conv1:(1x5x5+1)x6=156 parameters
conv2: (6x5x5+1)x16=2,416 parameters
fc1: (16×4×4+1)×120=30,840 parameters
fc2: (120+1)×84=10,164 parameters 
fc3: (84+1)x10=850 parameters

total parameters in LeNet-5 : 156 + 2,416+ 30,840+ 10,164+ 850 = 44,426

## Custom MLP Architecture::
Input: Flattened 1x28x28 image (784 inputs)

Hidden Layer 1: 500 units

Hidden Layer 2: 150 units

Output: 10 units (10 classes)

### Parameter Calculation for Custom MLP:
fc1: (784+1)x500 = 392,500 parameters
fc2: (500+1)x150= 75,150 parameters
fc3 : (150+1)x10 = 1,510 parameters

total parameters in Custom MLP : 392,500+75,150+1,510 = 469,160

