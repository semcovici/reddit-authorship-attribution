import sys
sys.path.append('../../src/')
from models.autorship import AuthorClassifier
from dataset.data_utils import temporal_train_test_split

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# External
import sys
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import matplotlib
from keras import Sequential
from keras.layers import Embedding, LSTM, Bidirectional, Dense, TextVectorization, GRU
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
import gc

## My lib
sys.path.append('../src/')
from dataset.data_utils import get_data, temporal_train_test_split
from models.models_utils import create_encoder,calculate_class_weight
from models.rnn_classifiers import RNNModel


RANDOM_SEED = 42


def create_model_and_generate_results(rnn_layer, encoder, authors_to_classify, metrics, X_train, y_train_enc, epochs, batch_size, validation_split,X_test, y_test,le,vectorization,classification_type,class_weight,i, input_type,input_shape):
    model = RNNModel(
        rnn_layer =rnn_layer,
        encoder=encoder, 
        output_shape = len(authors_to_classify), 
        metrics=metrics,
        input_type = input_type,
        input_shape= input_shape
        )
    
    model.fit(X_train,pd.get_dummies(y_train_enc), epochs = epochs, batch_size=batch_size, validation_split=validation_split, class_weight= class_weight)
    
    metrics_df = model.evaluate_model(X_test, y_test,le)
    
    metrics_df['rnn_layer'] = rnn_layer
    metrics_df['vectorizer'] = vectorization
    
    metrics_df.to_csv(f'../models/intermediate_results/{classification_type}/simple_{rnn_layer}_{vectorization}_part{i}.csv',index=False)
    
    del model, metrics_df

def multiclass_rnn_process(data, authors_to_classify,rnn_layer,metrics, epochs, batch_size, validation_split,vectorization,classification_type):
    i=0
    
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed = RANDOM_SEED)
    tf.random.set_seed(seed = RANDOM_SEED)
    
    X_train, X_test, y_train, y_test = temporal_train_test_split(
    data, authors_to_classify,
    train_size = 0.75
    )

    le = LabelEncoder().fit(y_train)

    y_train_enc = le.transform(y_train)
        
    class_weight = calculate_class_weight(y_train_enc)
    
    input_type = X_train.dtype
    
    if input_type == 'object':
        encoder = create_encoder(X_train)
        input_shape =  None
    elif input_type == 'float64':
        encoder = None,   
        input_shape =  (X_test.shape[1], X_test.shape[2])
    else:
        raise ValueError("Unsupported input_type: {}".format(input_type))

    
    create_model_and_generate_results(rnn_layer, encoder, authors_to_classify, metrics, X_train, y_train_enc, epochs, batch_size, validation_split,X_test, y_test,le,vectorization,classification_type,class_weight,i, input_type = input_type, input_shape=input_shape)
   
    del X_train, X_test, le, y_train, y_test
   
def one_vs_all_rnn_process(usernames, start_in,data,rnn_layer,metrics, epochs, batch_size, validation_split,vectorization,classification_type):
    
    for i, author in enumerate(usernames):
        
        if i >= start_in:
            
            print(f'author {author} ({i}) running ...')
            
            authors_to_classify = [author ,f'not_{author}']
                        
            keras.backend.clear_session()
            keras.utils.set_random_seed(seed = RANDOM_SEED)
            tf.random.set_seed(seed = RANDOM_SEED)
            
            X_train, X_test, y_train, y_test = temporal_train_test_split(
            data, usernames,
            train_size = 0.75
            )
            
            y_train = np.array([author if a == author else f'not_{author}' for a in y_train])
            
            y_test = np.array([author if a == author else f'not_{author}' for a in y_test])
        
            le = LabelEncoder().fit(y_train)

            y_train_enc = le.transform(y_train)
            
            class_weight = calculate_class_weight(y_train_enc)
            
            input_type = X_train.dtype
            
            if input_type == 'object':
                encoder = create_encoder(X_train)
                input_shape =  None
            elif input_type == 'float64':
                encoder = None,   
                input_shape =  (X_test.shape[1], X_test.shape[2])
            else:
                raise ValueError("Unsupported input_type: {}".format(input_type))
                        
            create_model_and_generate_results(rnn_layer, encoder, authors_to_classify, metrics, X_train, y_train_enc, epochs, batch_size, validation_split,X_test, y_test,le,vectorization,classification_type,class_weight,i,input_type,input_shape)
                                                                                                                    
            del encoder, X_train, X_test, le, y_train, y_test
            
            gc.collect()
        else:
            
            print(f'author {author} ({i}) was skipped')

    
def one_vs_one_rnn_process(data,usernames,start_in,rnn_layer,metrics, epochs, batch_size, validation_split,vectorization,classification_type):
    j = 0
    for i in range(len(usernames)):
        author1 = usernames.pop()

        
        for author2 in usernames:
            
            authors_to_classify = [author1 ,author2]
            
            if j >= start_in:
                
                print(f'combination {j} ({authors_to_classify}) running ...')
                keras.backend.clear_session()
                keras.utils.set_random_seed(seed = RANDOM_SEED)
                tf.random.set_seed(seed = RANDOM_SEED)
                
                X_train, X_test, y_train, y_test = temporal_train_test_split(
                data, authors_to_classify,
                train_size = 0.75
                )
                
            
                le = LabelEncoder().fit(y_train)

                y_train_enc = le.transform(y_train)
                
                
                input_type = X_train.dtype
                
                if input_type == 'object':
                    encoder = create_encoder(X_train)
                    input_shape =  None
                elif input_type == 'float64':
                    encoder = None,   
                    input_shape =  (X_test.shape[1], X_test.shape[2])
                else:
                    raise ValueError("Unsupported input_type: {}".format(input_type))
                
                class_weight = calculate_class_weight(y_train_enc)
                
                create_model_and_generate_results(rnn_layer, encoder, authors_to_classify, metrics, X_train, y_train_enc, epochs, batch_size, validation_split,X_test, y_test,le,vectorization,classification_type,class_weight,j, input_type,input_shape)
                                                                                                                        
                del encoder, X_train, X_test, le, y_train, y_test, authors_to_classify
                gc.collect()
            else: 
                
                print(f'combination {j} ({authors_to_classify}) was skipped')
            
            j+=1
        
        
    


def process_rnn(
data,
rnn_layer,
vectorization,
classification_type,
metrics,
start_in = 0,
epochs = 1000,
batch_size=128,
validation_split=0.1
):
    if isinstance(data, pd.DataFrame):
        usernames = list(np.unique(data["username"]))
    else:
        y = data[1]
        usernames = list(np.unique(y["username"]))
    
    if classification_type == 'multiclass':
        authors_to_classify = usernames
        multiclass_rnn_process(data, authors_to_classify,rnn_layer,metrics, epochs, batch_size, validation_split,vectorization,classification_type)
        gc.collect()
        
    elif classification_type == 'one-vs-all':
        one_vs_all_rnn_process(usernames, start_in,data,rnn_layer,metrics, epochs, batch_size, validation_split,vectorization,classification_type)
        gc.collect()
    elif classification_type == 'one-vs-one':
        one_vs_one_rnn_process(data,usernames,start_in,rnn_layer,metrics, epochs, batch_size, validation_split,vectorization,classification_type)
        gc.collect()



def process_sklearn(
    data, 
    clf, 
    vectorizer, 
    X_columns,
    scaler,
    classification_type,
    embeddings = False,
    sampling = None
    ):
    
    clf_str = clf.__str__()
    vect_str = vectorizer.__str__()
    
    evaluation = list()
    usernames = list(np.unique(data["username"]))

    if classification_type == 'multiclass':
        X_train, X_test, y_train, y_test = temporal_train_test_split(   
            data, usernames, return_dataframe=True)
        
        author_clf = AuthorClassifier(clf=clf, vectorizer=vectorizer, scaler=scaler, embeddings = embeddings,sampling = sampling)
        
        author_clf.fit(X_train.loc[:, X_columns], y_train)
        
        y_pred = author_clf.predict(X_test.loc[:, X_columns])
        
        metrics_dict = author_clf.evaluate(y_test, y_pred)
        metrics_dict.update({'classifier':clf_str})
        metrics_dict.update({'vectorizer':vect_str})
        
        evaluation.append(metrics_dict)
        
    elif classification_type == 'one-vs-one':
        for i in range(len(usernames)):
            author1 = usernames.pop()

            for author2 in usernames:
                X_train, X_test, y_train, y_test = temporal_train_test_split(   
                    data, [author1, author2], return_dataframe=True)

                author_clf = AuthorClassifier(clf=clf, vectorizer=vectorizer, scaler=scaler, embeddings = embeddings,sampling = sampling)
                
                author_clf.fit(X_train.loc[:, X_columns], y_train)
                
                y_pred = author_clf.predict(X_test.loc[:, X_columns])
                
                metrics_dict = author_clf.evaluate(y_test, y_pred)
                metrics_dict.update({'classifier':clf_str})
                metrics_dict.update({'vectorizer':vect_str})
                
                evaluation.append(metrics_dict)
                
    elif classification_type == 'one-vs-all':
        
        for i, author in enumerate(usernames):
            
            X_train, X_test, y_train, y_test = temporal_train_test_split(
            data, usernames,
            train_size = 0.75, return_dataframe=True
            )
            
            y_train = np.array([author if a == author else f'not_{author}' for a in y_train])
            
            y_test = np.array([author if a == author else f'not_{author}' for a in y_test])
            

            author_clf = AuthorClassifier(clf=clf, vectorizer=vectorizer, scaler=scaler, embeddings = embeddings,sampling = sampling)
            
            author_clf.fit(X_train.loc[:, X_columns], y_train)
            
            y_pred = author_clf.predict(X_test.loc[:, X_columns])
            
            metrics_dict = author_clf.evaluate(y_test, y_pred)
            metrics_dict.update({'classifier':clf_str})
            metrics_dict.update({'vectorizer':vect_str})
            
            evaluation.append(metrics_dict)
            
    else:
        raise ValueError("Unsupported classification_type: {}".format(classification_type))
 
    metrics = pd.DataFrame(evaluation)        
    return metrics

                    
