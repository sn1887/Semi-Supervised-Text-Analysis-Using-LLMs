# Load Libraries and Data
#--------------------------------------------------------------------------------------------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np, gc, re 
import pandas as pd 

from sklearn.datasets import fetch_20newsgroups

# Fetch the dataset
newsgroups = fetch_20newsgroups(subset='all')

# Inspecting the dataset
print("Categories:", newsgroups.target_names)
print("Number of  samples:", len(newsgroups.data))

train_id = [f'{i:05d}' for i in range(1, len(newsgroups.data) + 1)]

train = pd.DataFrame({'unique_id' : train_id , 'full_text': newsgroups.data, 'label': newsgroups.target})

# Split the Data
#--------------------------------------------------------------------------------------------------------------
from sklearn.model_selection import StratifiedKFold

FOLDS = 5
train["fold"] = -1
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
for fold,(train_index, val_index) in enumerate(skf.split(train,train["label"])):
    train.loc[val_index,"fold"] = fold
print('Train samples per fold:')
train.fold.value_counts().sort_index()

# Generate Embeddings
#--------------------------------------------------------------------------------------------------------------
from transformers import AutoModel,AutoTokenizer
import torch, torch.nn.functional as F
from tqdm import tqdm

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state.detach().cpu()
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )

class EmbedDataset(torch.utils.data.Dataset):
    def __init__(self,df,tokenizer,max_length):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max = max_length
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        text = self.df.loc[idx,"full_text"]
        tokens = self.tokenizer(
                text,
                None,
                add_special_tokens=True,
                padding='max_length',
                truncation=True,
                max_length=self.max,
                return_tensors="pt")
        tokens = {k:v.squeeze(0) for k,v in tokens.items()}
        return tokens

# Extract Embeddings
#--------------------------------------------------------------------------------------------------------------
def get_embeddings(model_name='', max_length=1024, batch_size=32, compute_train=True, compute_test=True):

    global train, test

    DEVICE = "cuda:1" # EXTRACT EMBEDDINGS WITH GPU #2
    path = "/kaggle/input/download-huggingface-models/"
    disk_name = path + model_name.replace("/","_")
    model = AutoModel.from_pretrained( disk_name , trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained( disk_name , trust_remote_code=True)

    ds_tr = EmbedDataset(train, tokenizer, max_length)
    embed_dataloader_tr = torch.utils.data.DataLoader(ds_tr,
                            batch_size=batch_size,
                            shuffle=False)
    ds_te = EmbedDataset(test, tokenizer, max_length)
    embed_dataloader_te = torch.utils.data.DataLoader(ds_te,
                            batch_size=batch_size,
                            shuffle=False)
    
    model = model.to(DEVICE)
    model.eval()

    # COMPUTE TRAIN EMBEDDINGS
    all_train_text_feats = []
    if compute_train:
        for batch in tqdm(embed_dataloader_tr,total=len(embed_dataloader_tr)):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):
                    model_output = model(input_ids=input_ids,attention_mask=attention_mask)
            sentence_embeddings = mean_pooling(model_output, attention_mask.detach().cpu())
            # Normalize the embeddings
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            sentence_embeddings =  sentence_embeddings.squeeze(0).detach().cpu().numpy()
            all_train_text_feats.extend(sentence_embeddings)
    all_train_text_feats = np.array(all_train_text_feats)

    # COMPUTE TEST EMBEDDINGS
    all_test_text_feats = []
    if compute_test:
        for batch in embed_dataloader_te:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):
                    model_output = model(input_ids=input_ids,attention_mask=attention_mask)
            sentence_embeddings = mean_pooling(model_output, attention_mask.detach().cpu())
            # Normalize the embeddings
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            sentence_embeddings =  sentence_embeddings.squeeze(0).detach().cpu().numpy()
            all_test_text_feats.extend(sentence_embeddings)
        all_test_text_feats = np.array(all_test_text_feats)
    all_test_text_feats = np.array(all_test_text_feats)

    # CLEAR MEMORY
    del ds_tr, ds_te
    del embed_dataloader_tr, embed_dataloader_te
    del model, tokenizer
    del model_output, sentence_embeddings, input_ids, attention_mask
    gc.collect()
    torch.cuda.empty_cache()

    # RETURN EMBEDDINGS
    return all_train_text_feats, all_test_text_feats

# EMBEDDINGS TO LOAD/COMPUTE
# PARAMETERS = (MODEL_NAME, MAX_LENGTH, BATCH_SIZE)
# CHOOSE LARGEST BATCH SIZE WITHOUT MEMORY ERROR

models = [
    ('microsoft/deberta-base', 1024, 32),
    ('microsoft/deberta-large', 1024, 8),
    ('microsoft/deberta-v3-large', 1024, 8),
    ('allenai/longformer-base-4096', 1024, 32),
    ('google/bigbird-roberta-base', 1024, 32),
    ('google/bigbird-roberta-large', 1024, 8),
]



path = "/kaggle/input/embeddings-newsgroup-atiml/"
all_train_embeds = []
all_test_embeds = []

for (model, max_length, batch_size) in models:
    name = path + model.replace("/","_") + ".npy"
    if os.path.exists(name):
        train_embed = np.load(name)
        print(f"Loading train embeddings for {name}")
    else:
        print(f"Computing train embeddings for {name}")
        train_embed, test_embed = get_embeddings(model_name=model, max_length=max_length, batch_size=batch_size, compute_train=True)
        np.save(name, train_embed)
    all_train_embeds.append(train_embed)

del train_embed


all_train_embeds = np.concatenate(all_train_embeds,axis=1)

gc.collect()
print('Our concatenated train embeddings have shape', all_train_embeds.shape )
