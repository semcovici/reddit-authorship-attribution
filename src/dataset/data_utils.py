from random import random
import pandas as pd
from sklearn.utils import shuffle
import praw
import os
import numpy as np
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score

def get_top_authors(df):
    n_comments = df.groupby("username")["comment"].count()
    top_authors = n_comments.index[(n_comments>=975) & (n_comments<=1025)]
    df = df[df.username.isin(top_authors)]
    return df

def get_data(csv_path, select_authors=True, remove_duplicates=True):
    df = pd.read_csv(csv_path)
    df = df[["username", "comment", "created_utc"]]
    df = shuffle(df, random_state=42)
    
    if select_authors: df = get_top_authors(df)
    if remove_duplicates: df.drop_duplicates("comment", inplace=True)
    return df

class SparseToArray():
    """
    https://stackoverflow.com/questions/28384680/scikit-learns-pipeline-a-sparse-matrix-was-passed-but-dense-data-is-required
    """

    def __repr__(self):
        return("SparseToArray()")

    def __str__(self):
        return("SparseToArray()")

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.toarray()
    

def temporal_train_test_split(df, list_authors, train_size = 0.75, return_dataframe = False):
    
    if isinstance(df, pd.DataFrame):
        data_2authors = df[df.username.isin(list_authors)]
        data_2authors = data_2authors.sort_values("created_utc")
        
        train_authors = []
        test_authors = []
        for author in list_authors:
            data_author1 = data_2authors[data_2authors.username == author]
            train_author = data_author1[:int(len(data_author1)*train_size)]
            test_author = data_author1[int(len(data_author1)*train_size):]
            
            train_authors.append(train_author.drop(["created_utc"], axis = 1))
            test_authors.append(test_author.drop(["created_utc"], axis = 1))
            
        X_train = shuffle(pd.concat(train_author.drop('username', axis = 1) for train_author in train_authors), random_state=42)
        X_train = X_train.squeeze()
        y_train = shuffle(pd.concat(train_author['username'] for train_author in train_authors), random_state=42)    
        
        X_test = shuffle(pd.concat(test_author.drop('username', axis = 1) for test_author in test_authors), random_state=42)
        X_test = X_test.squeeze()
        y_test = shuffle(pd.concat(test_author['username'] for test_author in test_authors), random_state=42)

        if not return_dataframe:
            
            X_train, X_test = X_train.values, X_test.values
            y_train, y_test = y_train.values, y_test.values
        else:
            pass

        return X_train, X_test, y_train, y_test
    
    elif isinstance(df, tuple):
        
        X = df[0]
        y = pd.concat([df[1], df[2]], axis = 1)
        data_2authors = y[y.username.isin(list_authors)]
        data_2authors = y.sort_values("created_utc")
        
        train_authors_indexes = []
        test_authors_indexes = []
        for author in list_authors:
            data_author1 = data_2authors[data_2authors.username == author]
            train_author_index = data_author1[:int(len(data_author1)*train_size)].index
            test_author_index = data_author1[int(len(data_author1)*train_size):].index
            
            train_authors_indexes.append(train_author_index)
            test_authors_indexes.append(test_author_index)
            

            
        X_train = shuffle(np.concatenate([X[train_author_index] for train_author_index in train_authors_indexes]), random_state=42)
        X_train = X_train.squeeze()
        y_train = shuffle(pd.concat(y.loc[train_author_index,'username'] for train_author_index in train_authors_indexes), random_state=42)    
        
        X_test = shuffle(np.concatenate([X[test_author_index] for test_author_index in test_authors_indexes]), random_state=42)
        X_test = X_test.squeeze()
        y_test = shuffle(pd.concat(y.loc[test_author_index,'username'] for test_author_index in test_authors_indexes), random_state=42)

        return X_train, X_test, y_train, y_test
    
    else:
        raise ValueError("Unsupported df type: {}".format(type(df)))

# def temporal_train_test_split(df, list_authors, 
                              
#                               train_size = 0.75,
#                               test_size = 0.25, 
#                               validation_size = 0):
    
#     if train_size + test_size + validation_size != 1: 
        
#         print('The "sizes" are incorrect')
        
#         return
    
    
#     data_2authors = df[df.username.isin(list_authors)]
#     data_2authors = data_2authors.sort_values("created_utc")
    
#     train_authors = []
#     test_authors = []
#     validation_authors = []
#     for author in list_authors:
        
#         data_author1 = data_2authors[data_2authors.username == author]
        
#         train_pos_end = int(len(data_author1)*train_size)
#         validation_pos_end = train_pos_end + int(len(data_author1)* validation_size)
                
#         train_author = data_author1[:train_pos_end]
                    
#         validation_author = data_author1[train_pos_end : validation_pos_end]
#         test_author = data_author1[validation_pos_end:]
#         validation_authors.append(validation_author.drop(["created_utc"], axis = 1))
            
        
#         train_authors.append(train_author.drop(["created_utc"], axis = 1))
#         test_authors.append(test_author.drop(["created_utc"], axis = 1))
        
        
#     X_train = shuffle(pd.concat(train_author.drop('username', axis = 1) for train_author in train_authors), random_state=42)
#     X_train = X_train.squeeze()
#     y_train = shuffle(pd.concat(train_author['username'] for train_author in train_authors), random_state=42)    
    
#     X_test = shuffle(pd.concat(test_author.drop('username', axis = 1) for test_author in test_authors), random_state=42)
#     X_test = X_test.squeeze()
#     y_test = shuffle(pd.concat(test_author['username'] for test_author in test_authors), random_state=42)
    
#     if validation_size > 0:
#         X_val = shuffle(pd.concat(validation_author.drop('username', axis = 1) for validation_author in validation_authors), random_state=42)
#         X_val = X_val.squeeze()
#         y_val = shuffle(pd.concat(validation_author['username'] for validation_author in validation_authors), random_state=42)
        
#         return X_train, X_test, X_val, y_train, y_test, y_val

#     return X_train, X_test, y_train, y_test