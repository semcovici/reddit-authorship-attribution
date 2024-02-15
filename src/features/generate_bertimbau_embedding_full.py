# %%
from transformers import AutoTokenizer  # Or BertTokenizer
from transformers import AutoTokenizer  # Or BertTokenizer
from transformers import BertModel
import torch
import tqdm
import sys
sys.path.append('../libs')
from utils import get_data 

import pandas as pd
import numpy as np
import datetime

# %%
max_length = 100
# Data Paths
data_input_path = '../../data/raw/authors.csv'
data_output_path = f'../../data/processed/bert-base-portuguese-cased_full__max_lenght={max_length}'

# %%
data = get_data(data_input_path)

# %%
bertimbau_base ='neuralmind/bert-base-portuguese-cased'
bertimbau_lg = 'neuralmind/bert-large-portuguese-cased'


model_name = bertimbau_base

# %%
data.head()

# %%
import datetime
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, BertModel
from tqdm import tqdm



# Auxiliary Functions
def bert_text_preparation_batch(texts, tokenizer):
    marked_texts = ["[CLS] " + t + " [SEP]" for t in texts]
    tokenized_texts = tokenizer.batch_encode_plus(marked_texts, truncation=True, max_length=max_length, padding='max_length', return_tensors='pt', return_attention_mask=True)
    return tokenized_texts

def get_bert_embeddings_batch(tokens_tensors, attention_mask, model):
    with torch.no_grad():
        outputs = model(input_ids=tokens_tensors, attention_mask=attention_mask)
        hidden_states = outputs[2][1:]

    token_embeddings = hidden_states[-1]
    token_embeddings = torch.squeeze(token_embeddings, dim=0)
    return token_embeddings.tolist()

# Import Models
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
model = BertModel.from_pretrained(model_name, output_hidden_states=True)

# Get Data
data_bert = data.copy()
emb_vector = []

# Parameters for Batch Processing
batch_size = 64
num_batches = (len(data_bert) + batch_size - 1) // batch_size

# Processing in Batches with tqdm
print(f'Start of Embedding. Datetime: {datetime.datetime.today()}')
for batch_num in tqdm(range(num_batches), desc="Processing Batches"):
    start_idx = batch_num * batch_size
    end_idx = min((batch_num + 1) * batch_size, len(data_bert))
    
    batch_data = data_bert.iloc[start_idx:end_idx]
    texts = batch_data['comment'].tolist()
    
    tokenized_texts = bert_text_preparation_batch(texts, tokenizer)
    tokens_tensor = tokenized_texts['input_ids']
    attention_mask = tokenized_texts['attention_mask']

    del tokenized_texts

    list_token_embeddings = get_bert_embeddings_batch(tokens_tensor, attention_mask, model)
    
    list_token_embeddings = np.array(list_token_embeddings)
        
    if len(emb_vector) == 0:
        emb_vector = list_token_embeddings
    else:
        emb_vector = np.concatenate([emb_vector, list_token_embeddings])
        
    del list_token_embeddings, tokens_tensor, attention_mask

print(f'End of Embedding. Datetime: {datetime.datetime.today()}')


# %%
np.save(data_output_path, emb_vector)

# %%
emb_vector.shape

# %%



