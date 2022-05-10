# import os
# os.chdir("../")
# print(os.getcwd())
from projecttools.utils import model_eval
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def test_model_eval():
    """
    This is test makes sure that all the values in the model metrics dataframe are between 0 and 1 (inclusive).
    Therefore ensuring that a) those values are numbers and b) they are between 0 and 1 as those metrics should be
    """
    
    data = pd.read_csv("data/adult.data",
                   names = ['age', 'workclass', 'fnlwgt', 'education','education-num',
                            'marital-status','occupation','relationship','race','sex',
                           'capital-gain','capital-loss','hours-per-week',
                            'native-country','income'])
    
    X = data[["age"]]
    y = data["income"].factorize()[0]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
    
    
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    
    train_preds = lr.predict(X_train)
    test_preds = lr.predict(X_test)
    
    pred_df = model_eval(y_train, y_test, train_preds, test_preds)
    
    
    assert pred_df.applymap(lambda x: 0 <= x <= 1).all().all()
    
    
    
    