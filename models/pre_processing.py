import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_data(path, label_column, normalize_data=False, exclude=[]):
    
    data = pd.read_csv(path,index_col = False)
    print(data.head())

    
    array = data.to_numpy()
    # exclude certain columns
    array = np.delete(array, exclude, axis=1)
    
    X, y = array[:, :label_column], array[:, label_column]
    
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
    
    if normalize_data:
        ss = StandardScaler()
        X_train = ss.fit_transform(X_train)
        X_test = ss.transform(X_test)
    
    return X_train, y_train, X_test, y_test, data.columns