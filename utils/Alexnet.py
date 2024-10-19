from google.colab import drive
drive.mount('/content/drive')


import os
import time
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# check GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset class
class ShapeStacksDataset(Dataset):
    def __init__(self, csv_file, img_folder, transform=None, test=False):
        self.data = pd.read_csv(csv_file)
        self.img_folder = img_folder
        self.transform = transform
        self.test = test  # 테스트용 플래그 추가

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_id = self.data.iloc[idx]['id']
        img_path = os.path.join(self.img_folder, f"{img_id}.jpg")
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image, img_id  # 테스트일 경우 id만 반환
        else:
            stable_height = self.data.iloc[idx]['stable_height']
            return image, torch.tensor(stable_height, dtype=torch.float32)


# image transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
}

# filepath
train_csv_path = '/content/drive/MyDrive/Colab Notebooks/COMP90086_2024_Project_train/train.csv'
test_csv_path = '/content/drive/MyDrive/Colab Notebooks/COMP90086_2024_Project_test/test.csv'
train_image_folder = '/content/drive/MyDrive/Colab Notebooks/COMP90086_2024_Project_train/train/'
test_image_folder = '/content/drive/MyDrive/Colab Notebooks/COMP90086_2024_Project_test/test/'

# load dataset
tr_ds = ShapeStacksDataset(train_csv_path, train_image_folder, transform=data_transforms['train'])
va_ds = ShapeStacksDataset(train_csv_path, train_image_folder, transform=data_transforms['test'])

# train/validate data
tr_dl = DataLoader(tr_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
va_dl = DataLoader(va_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)


# AlexNet model
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1)  # stable_height
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# train model
model = AlexNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()


# epochs and evaluation + early stopping
def train_model(model, train_loader, val_loader, epochs=50, patience=5):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_loss = float('inf')  # Start with an infinitely large value
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct_train += (outputs.round() == labels).sum().item()  # 정답 수 계산
            total_train += labels.size(0)

        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(correct_train / total_train)  # train accuracy 계산

        # 검증
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                correct_val += (outputs.round() == labels).sum().item()  # count the number of right anwers
                total_val += labels.size(0)

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(correct_val / total_val)

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, '
              f'Val Loss: {val_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, '
              f'Val Acc: {val_accuracies[-1]:.4f}')

        # Early Stopping
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

    return train_losses, val_losses, train_accuracies, val_accuracies


# evaluation
train_losses, val_losses, train_accuracies, val_accuracies = train_model(model, tr_dl, va_dl)

# save model
torch.save(model.state_dict(), 'alexnet_stable_height.pth')

# visualisation for model evaluation
import matplotlib.pyplot as plt

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    # Plotting Losses
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting Accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

# After training, call the plotting function
plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)


# test model
def test_model(model, test_loader, test_csv_path):
    model.eval()
    predictions = []
    ids = []
    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device)
            outputs = model(images)
            rounded_predictions = np.round(outputs.cpu().numpy().flatten())
            predictions.extend(rounded_predictions)
            ids.extend(image_ids)

    # export csv file
    test_df = pd.read_csv(test_csv_path)
    test_df['predicted_stable_height'] = predictions.astype(int)
    test_df[['id', 'predicted_stable_height']].to_csv('/content/drive/MyDrive/Colab Notebooks/test_predictions.csv', index=False)

# test dataset and dataloader
test_ds = ShapeStacksDataset(test_csv_path, test_image_folder, transform=data_transforms['test'], test=True)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# model load and test
model.load_state_dict(torch.load('alexnet_stable_height.pth'))
test_model(model, test_dl, test_csv_path)
