import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class GeneralTraniner:
    def __init__(self, model: nn.Module, 
                 dataset: Dataset, 
                 criterion: nn.Module, 
                 optimizer: optim.Optimizer, 
                 device: torch.device) -> None:
        self.model = model
        self.dataloader = self.loadData(dataset)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def loadData(self, batchSize: int = 32, 
                 isShuffle: bool = True,
                 numWorkers: int = 2, pinMemory: bool = False, dropLast: bool = False, 
                 timeout = 0, workerInitFn = None) -> None:
        return DataLoader(self.dataset , batch_size=batchSize, shuffle=isShuffle, 
                   num_workers=numWorkers, pin_memory=pinMemory, drop_last=dropLast, timeout=timeout, 
                   worker_init_fn=workerInitFn)

    def train(self, num_epochs: int) -> None:
        self.model.to(self.device)
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(self.train_loader)}')