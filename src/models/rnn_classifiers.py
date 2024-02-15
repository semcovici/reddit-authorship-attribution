# External
import sys
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Embedding, LSTM, Bidirectional, Dense, TextVectorization, GRU, SimpleRNN, Input
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score

RANDOM_SEED = 42

class RNNModel:
    def __init__(self, rnn_layer, output_shape, metrics, input_type = None,encoder = None, input_shape = None):
        
        self.rnn_layer = rnn_layer
        self.encoder = encoder
        self.output_shape = output_shape
        self.metrics = metrics
        self.loss_function = 'categorical_crossentropy'
        self.optimizer = keras.optimizers.Adam(1e-4)
        self.input_type =input_type
        
        keras.backend.clear_session()
        keras.utils.set_random_seed(seed = RANDOM_SEED)
        tf.random.set_seed(seed = RANDOM_SEED)

        
        model = Sequential()
        
        
        if input_type == 'object':
            
            layers_dim = 64
            
            model.add(self.encoder)
            model.add(Embedding(
                input_dim=len(self.encoder.get_vocabulary()),
                output_dim=layers_dim,
                mask_zero=True))         
            
        elif input_type == 'float64':
            
            layers_dim = 128
            
            model.add(Input(shape=input_shape))
            
        else:
            raise ValueError("Unsupported input_type: {}".format(input_type))
            

        
        if rnn_layer == 'Simple':
            model.add(SimpleRNN(layers_dim))
        elif rnn_layer == 'LSTM':
            model.add(LSTM(layers_dim))
        elif rnn_layer == 'GRU':
            model.add(GRU(layers_dim))
        elif rnn_layer == 'BISimple':
            model.add(Bidirectional(SimpleRNN(layers_dim)))
        elif rnn_layer == 'BILSTM':
            model.add(Bidirectional(LSTM(layers_dim)))
        elif rnn_layer == 'BIGRU':
            model.add(Bidirectional(GRU(layers_dim)))
        elif rnn_layer == 'double-BIGRU':
            model.add(Bidirectional(GRU(layers_dim, return_sequences=True)))
            model.add(Bidirectional(GRU(int(layers_dim/2))))
        elif rnn_layer == 'double-BILSTM':
            model.add(Bidirectional(LSTM(layers_dim, return_sequences=True)))
            model.add(Bidirectional(LSTM(int(layers_dim/2))))
        else:
            raise ValueError("Unsupported RNN layer type: {}".format(rnn_layer))
        
        model.add(Dense(layers_dim, activation='relu'))
        model.add(Dense(self.output_shape, activation='softmax'))

        model.compile(loss=self.loss_function,
                      optimizer=self.optimizer,
                      metrics=self.metrics)
        
        self.model = model
    
    def fit(
        self,
        X_train,
        y_train,
        epochs = 1,
        batch_size = 16,
        validation_split = 0.1,
        callbacks = [keras.callbacks.EarlyStopping(monitor='val_prc', patience=10, mode='max',restore_best_weights=True)],
        class_weight = None
    ):
        
        model = self.model

        history = model.fit(X_train,y_train, 
                            epochs=epochs,
                            validation_split=validation_split,
                            callbacks=callbacks,
                            batch_size=batch_size,
                            class_weight = class_weight) 
        
        self.model = model
        
    def evaluate_model(
        self,
        X_test,
        y_test,
        le #LabelEncoder
        ):
        
        
        y_pred_proba = self.model.predict(X_test)
        y_pred = y_pred_proba.argmax(axis =1)
        y_pred_labels = le.inverse_transform(y_pred) 
                
        usernames = le.classes_
        metrics = dict()
        
        if len(usernames) == 2:
            author1 = usernames[0]
            author2 = usernames[1]
            
            metrics[f"author1"] = author1
            metrics[f"author2"] = author2
            
            metrics[f"precision_author1"] = round(precision_score(y_test, y_pred_labels, pos_label=author1), 4)
            metrics[f"recall_author1"] = round(recall_score(y_test, y_pred_labels, pos_label=author1), 4)
            metrics[f"f1_score_author1"] = round(f1_score(y_test, y_pred_labels, pos_label=author1), 4)
            
            metrics[f"precision_author2"] = round(precision_score(y_test, y_pred_labels, pos_label=author2), 4)
            metrics[f"recall_author2"] = round(recall_score(y_test, y_pred_labels, pos_label=author2), 4)
            metrics[f"f1_score_author2"] = round(f1_score(y_test, y_pred_labels, pos_label=author2), 4)
            
            metrics["precision_weighted"] = round(precision_score(y_test, y_pred_labels, average='weighted'), 4 )
            metrics["precision_micro"] = round(precision_score(y_test, y_pred_labels, average='micro'), 4 )
            metrics["precision_macro"] = round(precision_score(y_test, y_pred_labels, average='macro'), 4 )
            metrics["recall_weighted"] = round(recall_score(y_test, y_pred_labels, average='weighted'), 4 )
            metrics["recall_micro"] = round(recall_score(y_test, y_pred_labels, average='micro'), 4 )
            metrics["recall_macro"] = round(recall_score(y_test, y_pred_labels, average='macro'), 4 )
            metrics["f1_weighted"] = round(f1_score(y_test, y_pred_labels, average='weighted'), 4 )
            metrics["f1_micro"] = round(f1_score(y_test, y_pred_labels, average='micro'), 4 )
            metrics["f1_macro"] = round(f1_score(y_test, y_pred_labels, average='macro'), 4 )
            metrics["auc_score"] = round(roc_auc_score(y_test, y_pred_proba[:,1]), 4)
            metrics["accuracy"] = round(accuracy_score(y_test, y_pred_labels), 4 )
            
            metrics_df = pd.DataFrame([metrics])
            
        else:
            
            usernames_encoded = [i for i in range(len(usernames))]
            
            for i in usernames_encoded:
                author = usernames[i]
                metrics[f"author{i}"] = author
                metrics[f"precision_author{i}"] = round(precision_score(y_test, y_pred_labels, average=None, labels=[author])[0], 4)
                metrics[f"recall_author{i}"] = round(recall_score(y_test, y_pred_labels, average=None, labels=[author])[0], 4)
                metrics[f"f1_score_author{i}"] = round(f1_score(y_test, y_pred_labels, average=None, labels=[author])[0], 4)

            metrics["precision_weighted"] = round(precision_score(y_test, y_pred_labels, average='weighted'), 4)
            metrics["precision_micro"] = round(precision_score(y_test, y_pred_labels, average='micro'), 4)
            metrics["precision_macro"] = round(precision_score(y_test, y_pred_labels, average='macro'), 4)
            metrics["recall_weighted"] = round(recall_score(y_test, y_pred_labels, average='weighted'), 4)
            metrics["recall_micro"] = round(recall_score(y_test, y_pred_labels, average='micro'), 4)
            metrics["recall_macro"] = round(recall_score(y_test, y_pred_labels, average='macro'), 4)
            metrics["f1_weighted"] = round(f1_score(y_test, y_pred_labels, average='weighted'), 4)
            metrics["f1_micro"] = round(f1_score(y_test, y_pred_labels, average='micro'), 4)
            metrics["f1_macro"] = round(f1_score(y_test, y_pred_labels, average='macro'), 4)
            
            if len(usernames) > 2:
                metrics["auc_score_ovr"] = round(roc_auc_score(y_test, y_pred_proba,multi_class='ovr' ),4)
            else:
                
                metrics["auc_score"] = round(roc_auc_score(y_test, y_pred_proba[:, 1]), 4)
            metrics["accuracy"] = round(accuracy_score(y_test, y_pred_labels), 4)

            metrics_df = pd.DataFrame([metrics])

        return metrics_df