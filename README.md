# LeNet5_MLP_project


# MNIST Digit Classification

This project implements digit classification on the MNIST dataset using the PyTorch framework. It includes two models: the classic LeNet-5 and a custom Multi-Layer Perceptron (MLP) designed to have a similar number of parameters as LeNet-5. The goal is to compare the performance of these models in terms of accuracy and loss over multiple training epochs.

## Project Structure

- `dataset.py`: Contains the MNIST dataset handling including loading and transforming images.
- `model.py`: Defines the architectures of LeNet-5 and CustomMLP.
- `main.py`: Main script to execute training and testing of the models.
- `README.md`: This file, explaining the project and how to run it.

## Setup

### Prerequisites

- Python 3.8+
- PyTorch 1.7+
- torchvision
- PIL
- argparse

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/<your-username>/mnist-classification.git
cd mnist-classification


## Running the Code

To run the training and testing for the models, execute:

```bash
python main.py --train_data_dir path/to/train/data --test_data_dir path/to/test/data

Models
LeNet-5
Architecture: Consists of two convolutional layers followed by three fully connected layers.
Activation: ReLU is used for non-linearity.
Pooling: MaxPooling is used.
Parameters: The model has a total of approximately X parameters.
CustomMLP
Architecture: A simple MLP with one hidden layer.
Activation: ReLU is used for non-linearity.
Parameters: The model has a total of approximately Y parameters, matching those of LeNet-5.
Training
Models were trained for 10 epochs using the SGD optimizer with a learning rate of 0.01 and momentum of 0.9. CrossEntropyLoss was used as the loss function.

Results
The performance of the models is summarized below:

Model	Train Accuracy	Test Accuracy	Train Loss	Test Loss
LeNet-5	XX.XX%	YY.YY%	Z.ZZZZ	A.AAAA
CustomMLP	XX.XX%	YY.YY%	Z.ZZZZ	A.AAAA
Plots
Include plots of training/testing loss and accuracy here (if available).





Discussion
Discuss the performance differences between the models and any observations from the loss and accuracy plots. Mention any challenges faced and how they were overcome.

Conclusions
Summarize the findings of this project, the effectiveness of the models, and any potential improvements that could be made.

Notes on Using GitHub
Create a Repository: Log in to GitHub, go to your profile or the main page, click on "New" to create a new repository. Name it appropriately based on your project.
Upload Files: You can upload files directly through the GitHub interface:
Navigate to your repository.
Click 'Add file' > 'Upload files'.
Drag your project files (dataset.py, model.py, main.py) into the space provided or use the file explorer to select them.
Commit the changes.
Edit README.md: You can edit README.md directly on GitHub:
Navigate to your repository.
Click on README.md.
Click the pencil icon to edit.
Paste the provided Markdown content.
Commit the changes.
This will make your project available on GitHub with a clear and comprehensive README. Adjust any specifics, such as adding actual parameter counts or replacing placeholder paths with real ones.
