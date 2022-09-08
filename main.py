from pytorch_lightning import Trainer
import torch
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger 

from models.mnist_network import LitMNIST

def mnist():
    model = LitMNIST(
        model_entity = "conv1",
        )
    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=3,
        callbacks=[TQDMProgressBar(refresh_rate=20)],
        logger=CSVLogger(save_dir="logs/mnist/"),
    )
    trainer.fit(model)


if __name__ == "__main__":
    mnist()
