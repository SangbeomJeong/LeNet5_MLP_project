import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
from dataset import MNIST
from model import *

def train(model, trn_loader, device, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for data, target in trn_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.size(0)
        _, predicted = torch.max(output, 1)
        correct += predicted.eq(target).sum().item()
        total += target.size(0)
    
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def test(model, tst_loader, device, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for data, target in tst_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output, 1)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def plot_metrics(train_losses, train_accuracies, test_losses, test_accuracies):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'r-', label='Training Loss')
    plt.plot(epochs, test_losses, 'b-', label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'r-', label='Training Accuracy')
    plt.plot(epochs, test_accuracies, 'b-', label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_testing_metrics.png')  # Save the figure as a PNG file
    plt.show()  # Display the figure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir', default="/home/sangbeom/sangbeom/class/train", type=str, required=False, help='directory path of the training images')
    parser.add_argument('--test_data_dir',  default="/home/sangbeom/sangbeom/class/test", type=str, required=False, help='directory path of the test images')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataset = MNIST(args.train_data_dir)
    test_dataset = MNIST(args.test_data_dir)
    trn_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    tst_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model = LeNet5_Regularized().to(device)  # Choose model here: CustomMLP or LeNet5
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    
    num_epochs = 10
    for epoch in range(num_epochs):
        trn_loss, trn_acc = train(model, trn_loader, device, criterion, optimizer)
        tst_loss, tst_acc = test(model, tst_loader, device, criterion)
        
        train_losses.append(trn_loss)
        train_accuracies.append(trn_acc)
        test_losses.append(tst_loss)
        test_accuracies.append(tst_acc)
        
        print(f'Epoch {epoch+1}: Train Loss {trn_loss:.4f}, Train Acc {trn_acc:.2f}%, Test Loss {tst_loss:.4f}, Test Acc {tst_acc:.2f}%')

    plot_metrics(train_losses, train_accuracies, test_losses, test_accuracies)

if __name__ == '__main__':
    main()


