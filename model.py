import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CustomMLP(nn.Module):
    def __init__(self):
        super(CustomMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 50)
        self.fc2 = nn.Linear(50, 30)
        self.fc3 = nn.Linear(30, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNet5_Regularized(nn.Module):
    def __init__(self):
        super(LeNet5_Regularized, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)  # Convolution layer
        self.bn1 = nn.BatchNorm2d(6)  # Batch normalization
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)  # Batch normalization
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Adjusted for batch norm and padding
        self.dropout1 = nn.Dropout(0.2)  # Dropout
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(0.2)  # Dropout
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Apply BN before activation
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Apply BN before activation
        x = x.view(-1, 16 * 5 * 5)  # Flatten the output for the dense layer
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Apply dropout after activation
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)  # Apply dropout after activation
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    model = LeNet5()
    total_params_LeNet5 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters in LeNet-5: ", total_params_LeNet5)

    model = CustomMLP()
    total_params_CustomMLP = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters in CustomMLP: ", total_params_CustomMLP)

    model = LeNet5_Regularized()
    total_params_LeNet5_Regularized = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters in CustomMLP: ", total_params_LeNet5_Regularized)








