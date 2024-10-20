---
title: "COMP90086 Final Project - Physical Reasoning"
output: html_notebook
---

## Overview
This project aims to verify the stability of stacking blocks using computer vision technology. The training and testing data are provided through the ShapeStacks dataset. The blocks come in four types (cubes, rectangular solids, spheres, cylinders), and the stack consists of between two and six blocks. The images were taken from various camera angles, backgrounds, and lighting conditions. The training dataset includes images and a CSV file that describes the height and characteristics of object stacks, and the task is to classify the stable height into one of six categories (1~6 levels).

The model uses a pre-trained ResNet152, which has been fine-tuned for this classification task. This document explains dataset preparation, model architecture, training process, and key hyperparameters used.

---
## Requirements
To run this code, the following libraries are required:

- Python 3.6+
- PyTorch
- torchvision
- pandas
- scikit-learn
- timm (for model creation)
- PIL (Pillow)

---
## Dataset Preparation
- The dataset consists of images of stacked objects and metadata (e.g., `stable_height`) contained in a CSV file.
- Images are stored in a specific directory and referenced using IDs from the CSV file.
- The dataset is split into training and validation sets in an 80:20 ratio using stratified sampling to maintain class balance.

### Dataset Class
- **ShapeStacksDataset**: This custom dataset class loads images and their corresponding labels. If the dataset is for testing, only the image and ID are returned.
- Images are resized to 224x224 and normalized using the mean and standard deviation values of ImageNet.

---
## Model Architecture
The model is based on ResNet152, a deep convolutional neural network (CNN) pre-trained on the ImageNet dataset. The original fully connected layer is replaced to match the number of output classes (six stable height categories).

### Key Modifications
- **Pre-trained ResNet152**: The model is created using `timm.create_model('resnet152', pretrained=True)`.
- **Modified Classifier**: The fully connected layer is modified to output six classes.

---
## Training Process
The training process involves the following steps:

1. **Data Augmentation**: Images are resized and normalized for consistency with the pre-trained ResNet152.
2. **Data Loading**: Training and validation data are loaded using PyTorch's DataLoader with a batch size of 8.
3. **Loss Function and Optimizer**:
   - **Loss Function**: CrossEntropyLoss is used for classification.
   - **Optimizer**: Adam optimizer with an initial learning rate of 0.0001.
4. **Learning Rate Scheduler**: StepLR scheduler reduces the learning rate by a factor of 0.1 every 5 epochs.
5. **Training Loop**:
   - The model is trained for a maximum of 20 epochs, with early stopping if validation accuracy does not improve for 10 epochs.
   - **Evaluation Metrics**: Accuracy, precision, recall, and F1-score are calculated for both training and validation sets.

---
## Model Evaluation and Saving
- **Metrics**: Precision, recall, F1-score, and accuracy are calculated during training to monitor model performance.
- **Early Stopping**: Training is stopped if validation accuracy does not improve for a specified number of epochs (patience = 10).
- **Model Saving**: The best model is saved to `resnet152_stable_height.pth` whenever validation accuracy improves.

---
## Usage
To train the model:

1. Ensure the dataset paths (`train_csv_path`, `train_image_folder`) are correctly set in the script.
2. Run the script to train the model.

---
## Key Hyperparameters
- **Batch Size**: 8
- **Learning Rate**: 0.0001
- **Number of Epochs**: 20
- **Early Stopping Patience**: 10
- **Learning Rate Scheduler**: StepLR, step size 5, gamma 0.1

---
## Results
Training and validation loss, accuracy, precision, recall, and F1-score are printed during training. The best model is saved whenever validation accuracy improves.
These metrics can be reviewed as a graph using the plt_metrics code.

---
## Testing the Model
- The model can be evaluated by running the test_model function to verify the final prediction results.
- Ensure that the dataset paths (test_csv_path, test_image_folder) are correctly set in the script.