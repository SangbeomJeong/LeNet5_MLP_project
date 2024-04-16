import os
import argparse
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class MNIST(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = os.listdir(data_dir)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        label = int(file_name.split('_')[1].split('.')[0])
        img_path = os.path.join(self.data_dir, file_name)
        img = Image.open(img_path).convert('L')
        img = self.transform(img)
        return img, label

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir', default="/home/sangbeom/sangbeom/class/train",  type=str, required=False, help='directory path of the training images')
    parser.add_argument('--test_data_dir',  default="/home/sangbeom/sangbeom/class/test", type=str, required=False, help='directory path of the test images')
    args = parser.parse_args()

    train_dataset = MNIST(args.train_data_dir)
    test_dataset = MNIST(args.test_data_dir)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    # Example of accessing a single sample
    img, label = train_dataset[0]
    print(f"Image shape: {img.shape}, Label: {label}")
