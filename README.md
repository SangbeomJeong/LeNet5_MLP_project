# LeNet5_MLP_project


# MNIST Digit Classification

This project implements digit classification on the MNIST dataset using the PyTorch framework. It includes two models: the classic LeNet-5 and a custom Multi-Layer Perceptron (MLP) designed to have a similar number of parameters as LeNet-5. The goal is to compare the performance of these models in terms of accuracy and loss over multiple training epochs.

## Project Structure

- `dataset.py`: Contains the MNIST dataset handling including loading and transforming images.
- `model.py`: Defines the architectures of LeNet-5 and CustomMLP.
- `main.py`: Main script to execute training and testing of the models.
- `README.md`: This file, explaining the project and how to run it.



#Model Implementations and Parameter Calculations
##LeNet-5 Architecture:

Input: 1x28x28 image (since MNIST images are 28x28 pixels and grayscale)
C1: Convolutional layer with 6 filters of size 5x5, stride 1 (output: 6x24x24)
S2: Max pooling layer with size 2x2, stride 2 (output: 6x12x12)
C3: Convolutional layer with 16 filters of size 5x5, stride 1 (output: 16x8x8)
S4: Max pooling layer with size 2x2, stride 2 (output: 16x4x4)
C5: Fully connected layer with 120 units
F6: Fully connected layer with 84 units
Output: Fully connected layer with 10 units (10 classes)
Parameter Calculation:

Conv1: 
(
1
×
5
×
5
+
1
)
×
6
=
156
(1×5×5+1)×6=156 parameters (weights + bias)
Conv2: 
(
6
×
5
×
5
+
1
)
×
16
=
2
,
416
(6×5×5+1)×16=2,416 parameters (weights + bias)
FC1: 
(
16
×
4
×
4
+
1
)
×
120
=
30
,
840
(16×4×4+1)×120=30,840 parameters (weights + bias)
FC2: 
(
120
+
1
)
×
84
=
10
,
164
(120+1)×84=10,164 parameters (weights + bias)
FC3: 
(
84
+
1
)
×
10
=
850
(84+1)×10=850 parameters (weights + bias)
Total Parameters in LeNet-5: 
156
+
2
,
416
+
30
,
840
+
10
,
164
+
850
=
44
,
426
156+2,416+30,840+10,164+850=44,426
Custom MLP
Architecture:

Input: Flattened 1x28x28 image (784 inputs)
Hidden Layer 1: 500 units
Hidden Layer 2: 150 units
Output: 10 units (10 classes)
Parameter Calculation:

FC1: 
(
784
+
1
)
×
500
=
392
,
500
(784+1)×500=392,500 parameters (weights + bias)
FC2: 
(
500
+
1
)
×
150
=
75
,
150
(500+1)×150=75,150 parameters (weights + bias)
FC3: 
(
150
+
1
)
×
10
=
1
,
510
(150+1)×10=1,510 parameters (weights + bias)
Total Parameters in Custom MLP: 
392
,
500
+
75
,
150
+
1
,
510
=
469
,
160
392,500+75,150+1,510=469,160
Discussion on Parameter Count
The LeNet-5 model has approximately 44,426 parameters, while the Custom MLP model, designed to be comparably complex, has 469,160 parameters. The substantial increase in parameters for the MLP model is primarily due to the first fully connected layer, which interacts directly with the flattened input image resulting in a large number of weights. This discrepancy highlights the efficiency of convolutional layers used in LeNet-5, which reduce the parameter count significantly by sharing weights across spatial hierarchies of the image.
