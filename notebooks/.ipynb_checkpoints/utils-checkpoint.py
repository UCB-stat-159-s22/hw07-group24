import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score



#Feature Engineering for v1 Model -- Kavin
'''Used for feature engineering data for v1 model (Kavin).
Requires df to be inputted with the following column names:
'age', 'workclass', 'fnlwgt', 'education', 'education-num',
'marital-status', 'occupation', 'relationship', 'race', 'sex',
'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
'income'
'''
def featureEngineeringKavinV1(df):
    #workclass
    dummies = pd.get_dummies(df["workclass"])
    df = pd.concat([df, dummies], axis=1)

    #education
    dummies = pd.get_dummies(df["education"])
    df = pd.concat([df, dummies], axis=1)

    #marital-status
    dummies = pd.get_dummies(df["marital-status"])
    df = pd.concat([df, dummies], axis=1)

    #occupation
    dummies = pd.get_dummies(df["occupation"])
    df = pd.concat([df, dummies], axis=1)

    #relationship
    dummies = pd.get_dummies(df["relationship"])
    df = pd.concat([df, dummies], axis=1)

    #race
    dummies = pd.get_dummies(df["race"])
    df = pd.concat([df, dummies], axis=1)

    #sex
    dummies = pd.get_dummies(df["sex"])
    df = pd.concat([df, dummies], axis=1)

    #native-country
    dummies = pd.get_dummies(df["native-country"])
    df = pd.concat([df, dummies], axis=1)
    
    #log transform age
    df['age log transformed'] = (df['age']+1).transform(np.log)
    
    #log transform education-num
    df['years in education log transformed'] = (df['education-num']+1).transform(np.log)
    
    df['capital-gain log transformed'] = (df['capital-gain']+1).transform(np.log)
    
    df['capital-loss log transformed'] = (df['capital-loss']+1).transform(np.log)
    
    #log transform hours-per-week
    df['hours-per-week log transformed'] = (df['hours-per-week']+1).transform(np.log)
    
    df["years educated / hours worked"] = df["years in education log transformed"] / df["hours-per-week log transformed"]

    df["capital gains * age"] = df["capital-gain log transformed"] * df["age log transformed"]

    #income
    dummies = pd.get_dummies(df["income"])
    df = pd.concat([df, dummies], axis=1)

    #drop original income column
    df.drop(['income'], axis=1, inplace=True)
    #drop >50K
    df = df.iloc[:, :-1]
    
    
    #dropping of columns that will be unused by model
    df.drop(['age', 'workclass', 'fnlwgt', 'education', 'education-num',
         'marital-status', 'occupation', 'relationship', 'race', 'sex',
         'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'], 
        axis = 1, 
        inplace = True)
    
    return df


def histboxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
    """
    Boxplot and histogram combined
    data: Pandas dataframe
    feature: Name of the dataframe for visualizing
    figsize: size of figure (default (12,7))
    kde: whether to show the density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # creating the 2 subplots
    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )  # boxplot will be created and a star will indicate the mean value of the column
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter"
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    )  # For histogram
    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram

    
def model_eval(train_labels, test_labels, train_preds, test_preds):
    """
    
    This will be a utils function that outputs a Pandas DataFrame of a models training and testing scores 
    for accuracy, precision, recall, and f1.
    
    
    Parameters
    ----------
    train_labels : 1d array-like object that represents the training data's target variables
    
    test_labels : 1d array-like object that represents the testing data's target variables
    
    train_preds : 1d array-like object that represents the predictions for the training data
    
    test_preds : 1d array-like object that represents the predictions for the testing data
    
    Returns
    -------
    
    pred_df : Pandas DataFrame of 4x2 dimenions.
    
    
    Examples
    --------
    
    >>>from utils import model_eval
    
    >>>model_eval(y_train, y_test, train_preds, test_preds)
    
    
                    Training | Testing
    Accuracy Score   0.37       0.35
    Precision Score  0.23       0.20
    Recall Score     0.30       0.28
    F1 Score         0.27.      0.24      
    """
    
    assert (len(train_labels) == len(train_preds)) and (len(test_labels) == len(test_preds)), "Mismatched dimensions in the parameters"
    
    
    metric_funcs = [accuracy_score, precision_score, recall_score, f1_score]
    
    train_scores = [func(train_labels, train_preds) for func in metric_funcs]
    test_scores = [func(test_labels, test_preds) for func in metric_funcs]
    
    
    pred_df = pd.DataFrame(data={"Training": train_scores, "Testing": test_scores},
                          index = ["Accuracy Score", "Precision Score", "Recall", "F1 Score"])
    return pred_df