from typing import Tuple
from pytorch_lightning import Trainer
import torch
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger 

from models.mnist_network import LitMNIST
from models.pre_processing import prepare_data
from models.training import train
from src.ALE import ale

#Importing Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.tree import ExtraTreeClassifier 

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
    
def tabular_data(model, path, label_column, ale_columns: Tuple):
    X_train, y_train, X_test, y_test = prepare_data(path, label_column)
    # available models are:
    
    model = train(model, X_train, y_train, X_test, y_test)

    ale(model,
        X_train,
        num_buckets=22,
        columns=ale_column)
    
    


if __name__ == "__main__":
    # available models are:
    
    # models = [
    #     KNeighborsClassifier(),
    #     SGDClassifier(),
    #     LogisticRegression(),
    #     RandomForestClassifier(),
    #     GradientBoostingClassifier(),
    #     AdaBoostClassifier(),
    #     BaggingClassifier(),
    #     SVC(),
    #     GaussianNB(),
    #     DummyClassifier(),
    #     ExtraTreeClassifier()
    #     ]
    model = AdaBoostClassifier()
    path = "datasets/heart_failure/heart_failure_clinical_records_dataset.csv"
    label_column = 12
    tabular_data(model,
                 path,
                 label_column,
                 (0, )
                 )
