from typing import Tuple
from pytorch_lightning import Trainer
from sklearn.neural_network import MLPClassifier, MLPRegressor
import torch
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger 

import pandas as pd

from models.mnist_network import LitMNIST
from models.pre_processing import prepare_data
from models.training import train
from src.ALE import ale
from foreign_ale import ale_plot

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
  
  
def tabular_data(model,
                 path,
                 label_column,
                 ale_columns: Tuple, 
                 normalize_data: bool = True,
                 exclude: list = []):
    
    X_train, y_train, X_test, y_test, columns = prepare_data(path,
                                                             label_column,
                                                             normalize_data,
                                                             exclude)
    
    model = train(model, X_train, y_train, X_test, y_test)

    # ale(model,
    #     X_train,
    #     grid_shape=(50,),
    #     columns=(3,))
    features = columns[:-1]
    print(features)
    ale_plot(model,
            pd.DataFrame(X_test, columns=features),
            ["temperature", "humidity"],
            bins=20,
            # monte_carlo=True,
            # monte_carlo_rep=100,
            # monte_carlo_ratio=0.6,
            )    
    
    
    # for i in range(12):
    #     ale_plot(model,
    #             pd.DataFrame(X_test, columns=features),
    #             pd.DataFrame(X_train, columns=features).columns[i],
    #             bins=20,
    #             monte_carlo=True,
    #             monte_carlo_rep=100,
    #             monte_carlo_ratio=0.6,
    #             )    

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
    #     ExtraTreeClassifier(),
    #     MLPClassifier(),
    #     MLPRegressor(),
    #     ]
    model = AdaBoostClassifier()
    model = RandomForestClassifier()
    model = MLPClassifier()
    model = MLPRegressor()
    path = "datasets/heart_failure/heart_failure_clinical_records_dataset.csv"
    path = "datasets/bike_rental/london_merged.csv"
    label_column = 1
    tabular_data(model,
                 path,
                 label_column,
                 ale_columns=(1, ),
                 exclude = [0]
                 )
