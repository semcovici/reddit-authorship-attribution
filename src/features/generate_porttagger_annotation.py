import logging 
import os 
from pathlib import Path 
from typing import List, Tuple 
import sys 
sys.path.append('../data/')
from data_utils import get_data

import pandas as pd 
import spacy 
import torch 
from dante_tokenizer import DanteTokenizer 
from transformers import AutoModelForTokenClassification, AutoTokenizer 
import contractions
from tqdm import tqdm

try: 
    nlp = spacy.load("pt_core_news_sm") 
except Exception: 
    os.system("python -m spacy download pt_core_news_sm") 
    nlp = spacy.load("pt_core_news_sm") 
dt_tokenizer = DanteTokenizer() 
 
# Default model information
default_model = "Tweets (stock market)" 
model_choices = { 
    "news-base": "Emanuel/porttagger-news-base", 
    "tweets-base": "Emanuel/porttagger-tweets-base", 
    "oilgas-base": "Emanuel/porttagger-oilgas-base", 
    "base": "Emanuel/porttagger-base", 
} 

# Pre-tokenizers for different models
pre_tokenizers = { 
    "news-base": nlp, 
    "tweets-base": dt_tokenizer.tokenize, 
    "oilgas-base": nlp, 
    "base": nlp, 
} 

# Logger setup
logger = logging.getLogger() 
logger.setLevel(logging.DEBUG)

def expand_contractions(text):
    """Expand shortened words, e.g., don't to do not"""
    return contractions.fix(text)

def predict(pre_tokenizer, tokenizer, model, text, logger=None) -> Tuple[List[str], List[str]]: 
    doc = pre_tokenizer(text) 
    tokens = [token.text if not isinstance(token, str) else token for token in doc] 
 
    logger.info("Starting predictions for sentence: {}".format(text)) 
 
    input_tokens = tokenizer( 
        tokens, 
        return_tensors="pt", 
        is_split_into_words=True, 
        return_offsets_mapping=True, 
        return_special_tokens_mask=True,
        max_length=512,
        truncation=True
    ) 
    output = model(input_tokens["input_ids"]) 
 
    i_token = 0 
    labels = [] 
    scores = [] 
    for off, is_special_token, pred in zip( 
        input_tokens["offset_mapping"][0], 
        input_tokens["special_tokens_mask"][0], 
        output.logits[0], 
    ): 
        if is_special_token or off[0] > 0: 
            continue 
        label = model.config.id2label[int(pred.argmax(axis=-1))] 
        if logger is not None: 
            logger.info("{}, {}, {}".format(off, tokens[i_token], label)) 
        labels.append(label) 
        scores.append("{:.2f}".format(100 * float(torch.softmax(pred, dim=-1).detach().max()))) 
        i_token += 1 
 
    return tokens, labels, scores

def batch_analysis(pre_tokenizer, tokenizer, model, df, id_label, text_label): 
    ids = df[id_label] 
    texts = df[text_label] 

    df_text_pos = pd.DataFrame({
        'username': [],
        'comment': [],
        'created_utc': [],
        'pos': []
    })
    
    for id, text in tqdm(zip(ids, texts), total=len(ids)): 
        text = expand_contractions(text) 
        tokens, labels, scores = predict(pre_tokenizer, tokenizer, model, text, logger) 
        labels_string = ' '.join(labels)
        
        new_row = {
            'username': df.loc[id, 'username'],
            'comment': df.loc[id, 'comment'],
            'created_utc': df.loc[id, 'created_utc'],
            'pos': labels_string  
        }
        
        df_text_pos.loc[len(df_text_pos)] = new_row

    return df_text_pos

def main():
    data_input_path = "../../data/raw/authors.csv"
    txt_column = "comment"
    data = get_data(data_input_path)
    index = data.reset_index().pop('index')
    data = data.reset_index(drop=True).reset_index()
    id_label = 'index' 
    text_label = 'comment'

    for model_name, model_path in model_choices.items():
        print(f'Running: {model_name}')
        
        model = AutoModelForTokenClassification.from_pretrained(model_path) 
        tokenizer = AutoTokenizer.from_pretrained(model_path) 
        pre_tokenizer = pre_tokenizers[model_name]

        data_output_path_conllu = f'../../data/processed/authors_porttagger-{model_name}.csv'
        
        labels = batch_analysis(pre_tokenizer, tokenizer, model, data, id_label, text_label)
        labels.to_csv(data_output_path_conllu, index=False)
        
        print(f'End')

if __name__ == '__main__':
    main()