from keras.layers import  TextVectorization
import numpy as np


#https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization
def create_encoder(
    texts,
    output_mode = 'int',
    vocab = 1000,
    standardize=None,
    output_sequence_length=512,
    pad_to_max_tokens=True,
    ):
    
    encoder = TextVectorization(
        output_mode = output_mode,
        max_tokens=vocab,
        output_sequence_length=output_sequence_length,
        pad_to_max_tokens=pad_to_max_tokens,
        standardize=standardize
        )

    encoder.adapt(texts)
    
    return encoder

# https://www.tensorflow.org/tutorials/structured_data/imbalanced_data?hl=pt-br#calculate_class_weights
def calculate_class_weight(y):
    
    class_weights = {}
    
    classes = np.unique(y)
    
    
    for cl in classes:
        
        count_class = len(y[y==cl])
        
        weight = (1/len(y[y==cl])) * (len(y)/len(classes))
    
        class_weights.update({cl: weight})
        
    return class_weights