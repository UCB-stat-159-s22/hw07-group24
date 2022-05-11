import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn import preprocessing

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
    
    #log transform hours-per-week
    df['hours-per-week log transformed'] = (df['hours-per-week']+1).transform(np.log)
    
    #log transform capital-gain
    df['capital-gain log transformed'] = (df['capital-gain']+1).transform(np.log)
    
    #log transform capital-loss
    df['capital-loss log transformed'] = (df['capital-loss']+1).transform(np.log)
    
    
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

def feature_engineering_winston(data):
    for colname in data:
        types = data.dtypes.to_dict()
        check = []
        if len(pd.unique(data[colname])) > 10:
            check.append(colname)
        if str(types[colname]) =="object":
            data[colname] = pd.factorize(data[colname])[0]
    for colname in data:
        if data[colname].mean() > 1000:
            data[colname] = np.log(data[colname] + 1)
    income = data["income"]
    data.drop("income", axis = 1, inplace=True)
    scaler = StandardScaler()
    tmp = scaler.fit_transform(data)
    df = pd.DataFrame(index = data.index, data=tmp, columns = data.columns)
    data = pd.concat([df, income], axis = 1)
    return data

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



def feat_eng_split(features, target, split=0.25):
    """
    This is an utility function that creates a feature engineering pipeline which fits on and transform the training data,
    followed by transforming the testing data to its rules.
    
    The pipeline normalizes the numerical data and one hot encodes the categorical data. This function automatically detects which 
    features are numerical and which are categorical so it knows which columns to standardize and which to one hot encode.
    
    The pipeline also label encodes the target so that it converts its categories to numbers.
    
    
    Parameters
    ----------
    features : A 2d Pandas dataframe of the independent variables
    
    target : A 1d like array of the target variable
    
    split : The train-test-split ratio, defaults to 25%
    
    
    
    Returns
    ----------
    
    X_train_fe : The transformed 2d matrix of the training data features
    
    X_test_fe : The transformed 2d matrix of the testing data features
    
    y_train_fe : The transformed 1d array of the training target variable
    
    y_test_fe : The transformed 1d array of the testing target variable
    
    
    
    Examples
    --------
    
    >>>from utils import feat_eng_split
    
    >>>feat_eng_split(y_train, y_test, train_preds, test_preds)
    
    
    """
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=split, stratify=target, random_state=1)
    
    le = LabelEncoder()
    y_train_le = le.fit_transform(y_train)
    y_test_le = le.transform(y_test)
    
    
    feat_cols = X_train.columns
    transformer_steps = []
    
    num_cols = features.select_dtypes("number").columns.tolist()
    num_cols_mask = feat_cols.isin(num_cols)
        
    pipeline_num = Pipeline([("scale", StandardScaler())])
    column_transform_scale_step = ("numerical", pipeline_num, num_cols_mask)
    transformer_steps.append(column_transform_scale_step)
    
        
    cat_cols = features.select_dtypes("object").columns.tolist()
    cat_cols_mask= feat_cols.isin(cat_cols)

    pipeline_cat = Pipeline([("ohe", OneHotEncoder(categories='auto', 
                                                   handle_unknown='error', 
                                                   sparse=False, drop="first"))])
    column_transform_ohe_step = ("cat", pipeline_cat, cat_cols_mask)

    transformer_steps.append(column_transform_ohe_step)
        
    feat_eng_pipe = ColumnTransformer(transformers=transformer_steps)
    
    X_train_fe = feat_eng_pipe.fit_transform(X_train)
    X_test_fe = feat_eng_pipe.transform(X_test)
    
    
    ohe_col_names = feat_eng_pipe.named_transformers_["cat"]["ohe"].get_feature_names(cat_cols).tolist()
    
    column_names = num_cols + ohe_col_names
    
    X_train_fe = pd.DataFrame(index=X_train.index, data = X_train_fe, columns=column_names)
    X_test_fe = pd.DataFrame(index=X_test.index, data = X_test_fe, columns=column_names)
    

    return X_train_fe, X_test_fe, y_train_le, y_test_le


# def save_model(model, path):
    
#     joblib.dump(model, path)
    