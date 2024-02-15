# install the tagger with:
# python -m spacy download pt_core_news_lg
# python -m spacy download pt_core_news_md
# python -m spacy download pt_core_news_sm
import sys
import spacy
import numpy as np
from tensorflow import keras
from tqdm import tqdm
from spacy.lang.pt.examples import sentences 

import sys
sys.path.append('../../src')
from dataset.data_utils import get_data

def create_pos_column(data, model_name, txt_column, seed=None):
    if seed is not None:
        spacy.util.fix_random_seed(seed)
    
    pos_converter = spacy.load(model_name)
    pos_texts = [pos_converter(text) for text in tqdm(data[txt_column])]

    pos = np.empty(len(pos_texts), dtype='object')

    for i in tqdm(range(len(pos_texts))):
        pos[i] = " ".join([token.pos_ for token in pos_texts[i]])
        
    data["pos"] = pos
    
    return data

def main():
    data_input_path = "../../data/raw/authors.csv"
    txt_column = "comment"
    models_names = ["pt_core_news_lg", "pt_core_news_sm", "pt_core_news_md"]
    
    data = get_data(data_input_path) 
    
    for model_name in models_names:
        
        print(f'Running: {model_name}')
        
        data_output_path = f"../../data/processed/authors_{model_name}.csv"
        
        # Set seed for reproducibility 
        seed = 42
        data_pos = create_pos_column(data, model_name, txt_column, seed)
        
        data_pos.to_csv(data_output_path) 
        
        
if __name__ == '__main__':
    main()
