# LeNet5_MLP_project


# MNIST Digit Classification

This project implements digit classification on the MNIST dataset using the PyTorch framework. It includes two models: the classic LeNet-5 and a custom Multi-Layer Perceptron (MLP) designed to have a similar number of parameters as LeNet-5. The goal is to compare the performance of these models in terms of accuracy and loss over multiple training epochs.

## Code Configuration

- `dataset.py`: Contains the MNIST dataset handling including loading and transforming images.
- `model.py`: Defines the architectures of LeNet-5 and CustomMLP.
- `main.py`: Main script to execute training and testing of the models.
- `README.md`: This file, explaining the project and how to run it.

## How to run it 
```
### code download
git clone https://github.com/SangbeomJeong/LeNet5_MLP_project.git

### run 
CUDA_VISIBLE_DEVICES=0 python main.py
```

# Model Implementations and Parameter Calculations
## LeNet-5 Architecture:

Input: 1x28x28 image (MNIST images are 28x28 pixels and grayscale)

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
Input: Flattened 1x28x28 image (MNIST images are 28x28 pixels and grayscale)

Hidden Layer 1: 50 units

Hidden Layer 2: 30 units

Output: 10 units (10 classes)

### Parameter Calculation for Custom MLP:
fc1: (784+1)x50 = 39,250 parameters

fc2: (50+1)x30= 1,530 parameters

fc3 : (30+1)x10 = 310 parameters

total parameters in Custom MLP : 39,250 + 1,530 + 310 = 41,090

## Compare parameters numbers 


|   Architecture        |   total parameters    |
|-----------------------|-----------------------|
|     LeNet-5           |    44,426             |
|    Custom MLP         |    41,090             | 


# Performance Comparison
In this project, we evaluated two models on the MNIST dataset: the classic LeNet-5 and a custom-designed Multi-Layer Perceptron (MLP) that has a similar number of parameters as LeNet-5. The goal was to compare their performance in terms of accuracy and loss across multiple training epochs.

## Model Performance Summary
### LeNet-5

Training Accuracy: 99.60%
Testing Accuracy: 98.84%
Training Loss: 0.0121
Testing Loss: 0.0398

![Uploading training_testing_metrics.png…](figures/LeNet5_train_test_result.png)

### Custom MLP

Training Accuracy: 98.86%
Testing Accuracy: 97.26%
Training Loss: 0.0356
Testing Loss: 0.0992

![Uploading training_testing_metrics.png…](figures/customMLP_train_test_result.png)


## Analysis and Discussion
The LeNet-5 model generally exhibits higher accuracy compared to the custom MLP model. This superiority can be attributed to the convolutional layers of LeNet-5, which are effective at learning spatial hierarchies in images. convolutional networks can extract significant spatial features, outperforming the relatively complex MLP model in tasks such as image classification.

The accuracy of LeNet-5 on the test dataset should align closely with known benchmarks (e.g., around 98%).



## Conclusion
In this study, the LeNet-5 model demonstrated performance close to known benchmarks and achieved lower losses and higher accuracies than the custom MLP model. These results suggest that convolutional neural networks like LeNet-5 are more suited for tasks involving image classification, such as MNIST digit classification.


# Regularization Techniques to Improve LeNet-5 Model
In this project, we aim to enhance the performance of the LeNet-5 model on the MNIST digit classification task. Regularization techniques are crucial for improving generalization, reducing overfitting, and achieving higher accuracy, especially when working with a limited amount of training data. The following regularization techniques were employed:

## Dropout
Dropout is a regularization technique that prevents neural networks from overfitting. It works by randomly setting the output features of hidden units to zero at each update during training time, which helps to break up happenstance correlations that can form in the weights.

Implementation: Dropout was added after each fully connected layer in the LeNet-5 architecture. A dropout rate of 0.2 was used, meaning each neuron was kept with a probability of 0.2 during the training phase.

## Batch Normalization
Batch Normalization is used to stabilize the neural network training by normalizing the layer inputs. It smoothens the optimization landscape and accelerates the training process.

Implementation: Batch normalization layers were added after each convolutional layer in the LeNet-5 model. This was done before applying the activation function to ensure the data scale and distribution remains normalized throughout training.

## Results and Discussion
These regularization techniques combined showed a significant improvement in model performance. The inclusion of dropout and batch normalization helped in reducing overfitting and making the network less sensitive to the specific weights of neurons. Data augmentation expanded the variety of training examples, which bolstered the model's ability to generalize to new, unseen data.

Performance Metrics: The table below shows the accuracy improvements before and after regularization:

![Uploading training_testing_metrics.png…](figures/LeNet5_Regularized_train_test_result.png)

Metric	Before Regularization	After Regularization

| Metric                | Before Regularization | After Regularization |
|-----------------------|-----------------------|----------------------|
| Test Accuracy         | 98.84%                | 99.03%               |



These results indicate that the regularization techniques effectively enhanced the model's generalization capabilities, as evidenced by the improved validation accuracy.










