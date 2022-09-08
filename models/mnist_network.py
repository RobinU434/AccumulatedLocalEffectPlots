import os

import pandas as pd
import seaborn as sn
import torch
from IPython import display
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64


class LitMNIST(LightningModule):
    def __init__(self, data_dir=PATH_DATASETS, hidden_size=64, learning_rate=2e-4, model_entity="linear"):
        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        # Define PyTorch model
        self.model = self.load_model(entity = model_entity,
                                     channels = channels, 
                                     width = width,
                                     height = height,
                                     hidden_size = hidden_size)

        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()
        
    def load_model(self, entity: str, **kwargs):
        
        if entity == "linear":
            channels = kwargs.pop("channels")
            width = kwargs.pop("width")
            height = kwargs.pop("height")
            hidden_size = kwargs.pop("hidden_size")
            
            # Define PyTorch model
            model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(channels * width * height, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, self.num_classes),
            )
        elif entity == "conv1":
            model = nn.Sequential(
                nn.Conv2d(
                    in_channels=1,              
                    out_channels=16,            
                    kernel_size=5,              
                    stride=1,                   
                    padding=2,
                    ),                              
                nn.ReLU(),                      
                nn.MaxPool2d(kernel_size=2),
                nn.Flatten(),
                nn.Linear(32 * 7 * 7 * 2, 10) # flatten the output of conv2 to (batch_size, 32 * 7 * 7) 
                )
        elif entity == "conv2":
            model = nn.Sequential(         
                nn.Conv2d(16, 32, 5, 1, 2),     
                nn.ReLU(),                      
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(32 * 7 * 7 * 2, 10)  # fully connected layer, output 10 classes                
            ) 
            
        elif entity == "conv3":
            model = nn.Sequential( 
                nn.Conv2d(1, 32, kernel_size=5),
                nn.Conv2d(32, 32, kernel_size=5),
                nn.Conv2d(32,64, kernel_size=5),
                nn.Linear(3*3*64, 256),
                nn.Linear(256, 10),
            )
        else:
            raise NotImplementedError("choose either 'linear', 'conv1' or 'conv2'")

        print(model)
        return model

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=BATCH_SIZE)