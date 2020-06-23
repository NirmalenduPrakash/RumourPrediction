
import pandas as pd
import pandas as pd
import torch
from transformers import BertTokenizer
from torch.nn import functional as F    
from transformers import BertModel
import nltk
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
import tqdm
import os
import json
import random

LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    "hidden": 0
}
def load_nli_data(path, snli=False):
    """
    Load MultiNLI or SNLI data.
    If the "snli" parameter is set to True, a genre label of snli will be assigned to the data. 
    """
    data = []
    with open(path) as f:
        for line in f:
            loaded_example = json.loads(line,strict=False)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue
            loaded_example["label"] = LABEL_MAP[loaded_example["gold_label"]]
            if snli:
                loaded_example["genre"] = "snli"
            data.append(loaded_example)
        random.seed(1)
        random.shuffle(data)
    return data

data=load_nli_data('/home/svu/e0401988/NLP/BERT_NLI/multinli_1.0/multinli_1.0_train.jsonl')
data=data[:20000]

labels={'neutral':0,'contradiction':1,'entailment':2}
train_data=pd.DataFrame(data)
# train_data.head()

train_data.drop([label for label in train_data.columns if label not in ['gold_label','sentence1','sentence2']],axis=1,inplace=True)
train_data['gold_label']=train_data['gold_label'].apply(lambda x:labels[x])
# train_data.head()

# bert_model=BertModel.from_pretrained('bert-base-uncased')
# bert_model.save_pretrained('/content/drive/My Drive/multinli_1.0')
tokenizer = BertTokenizer.from_pretrained('/home/svu/e0401988/NLP/BERT_NLI/BERT_CONFIG')

def pad(tokens):
  if(len(tokens)<512):
    return tokens+['[PAD]' for _ in range(512-len(tokens))]
  else:
    return tokens[:512] 

train_data['s1_encoding']=train_data['sentence1'].apply(lambda x: ['[CLS]'] + tokenizer.tokenize(x))
train_data['s2_encoding']=train_data['sentence2'].apply(lambda x: ['[CLS]'] + tokenizer.tokenize(x))
  
train_data['s1_encoding']=train_data['s1_encoding'].apply(lambda x: pad(x))
train_data['s2_encoding']=train_data['s2_encoding'].apply(lambda x: pad(x))

train_data['s1_encoding']=train_data['s1_encoding'].apply(lambda x:tokenizer.convert_tokens_to_ids(x))
train_data['s2_encoding']=train_data['s2_encoding'].apply(lambda x:tokenizer.convert_tokens_to_ids(x))

train_data['attn_mask1'] = train_data['s1_encoding'].apply(lambda x:[0 if token==0 else 1 for token in x])
train_data['attn_mask2'] = train_data['s2_encoding'].apply(lambda x:[0 if token==0 else 1 for token in x])

train_data.head()

# max([len(x) for x in list(train_data['s1_encoding'])])

# tokenizer.convert_tokens_to_ids('[PAD]')

# with open('/content/drive/My Drive/multi_nli_train_data.pkl','wb') as f:
#   pickle.dump(train_data,f)

val_data=load_nli_data('/home/svu/e0401988/NLP/BERT_NLI/multinli_1.0/multinli_1.0_dev_matched.jsonl')
val_data.extend(load_nli_data('/home/svu/e0401988/NLP/BERT_NLI/multinli_1.0/multinli_1.0_dev_mismatched.jsonl'))
val_data=val_data[:5000]
val_data=pd.DataFrame(val_data)
val_data.drop([label for label in val_data.columns if label not in ['gold_label','sentence1','sentence2']],axis=1,inplace=True)
val_data['gold_label']=val_data['gold_label'].apply(lambda x:labels[x])

val_data['s1_encoding']=val_data['sentence1'].apply(lambda x: ['[CLS]'] + tokenizer.tokenize(x))
val_data['s2_encoding']=val_data['sentence2'].apply(lambda x: ['[CLS]'] + tokenizer.tokenize(x))  
val_data['s1_encoding']=val_data['s1_encoding'].apply(lambda x: pad(x))
val_data['s2_encoding']=val_data['s2_encoding'].apply(lambda x: pad(x))
val_data['s1_encoding']=val_data['s1_encoding'].apply(lambda x:tokenizer.convert_tokens_to_ids(x))
val_data['s2_encoding']=val_data['s2_encoding'].apply(lambda x:tokenizer.convert_tokens_to_ids(x))
val_data['attn_mask1'] = val_data['s1_encoding'].apply(lambda x:[0 if token==0 else 1 for token in x])
val_data['attn_mask2'] = val_data['s2_encoding'].apply(lambda x:[0 if token==0 else 1 for token in x])

# val_data.head()

class ClassifierDataset(Dataset):
  def __init__(self,df):
    self.df=df

  def __len__(self): 
    return len(self.df)

  def __getitem__(self,index):
    return torch.tensor(self.df.iloc[index]['s1_encoding']),torch.tensor(self.df.iloc[index]['s2_encoding'])\
        ,torch.tensor(self.df.iloc[index]['attn_mask1']),torch.tensor(self.df.iloc[index]['attn_mask2']),torch.tensor(self.df.iloc[index]['gold_label'])

# def get_data():

class SentenceClassifier(nn.Module):
    def __init__(self):
        super(SentenceClassifier, self).__init__()
        self.bert_layer = BertModel.from_pretrained('/home/svu/e0401988/NLP/BERT_NLI/BERT_CONFIG')
        self.linear = nn.Linear(768*2, 3)
        self.softmax = nn.Softmax()

    def forward(self, seq1,seq2, mask1,mask2):
        _,cls1 = self.bert_layer(seq1, attention_mask = mask1)
        _,cls2 = self.bert_layer(seq2, attention_mask = mask2)
        
        output=torch.cat((cls1,cls2),dim=-1)
        return self.softmax(self.linear(output))

train_set=ClassifierDataset(train_data)
val_set=ClassifierDataset(val_data)
train_loader=DataLoader(train_set, batch_size = 16)
val_loader = DataLoader(val_set, batch_size = 16)        

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model=SentenceClassifier().to(device)
model.load_state_dict(torch.load('/home/svu/e0401988/NLP/BERT_NLI/multinli_model.pt'))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 2e-6)

# training_loss=[]
val_losses=[]

for _e in range(5):
    train_loss=0
    for t, (seq1,seq2,mask1,mask2,labels) in enumerate(train_loader):
        seq1=seq1.to(device)
        seq2=seq2.to(device)
        mask1=mask1.to(device)
        mask2=mask2.to(device)
        labels =labels.to(device) #torch.tensor(data_batch).to(device)
                
        optimizer.zero_grad()
        logits=model(seq1,seq2,mask1,mask2)
        loss = criterion(logits.squeeze(-1), labels.long())
        train_loss+=loss.data.item()
        loss.backward()
        optimizer.step()
    print(train_loss)    
    val_loss=0
    with torch.no_grad():
      for t, (seq1,seq2,mask1,mask2,labels) in enumerate(val_loader):
        seq1=seq1.to(device)
        seq2=seq2.to(device)
        mask1=mask1.to(device)
        mask2=mask2.to(device)
        labels =labels.to(device)
        logits=model(seq1,seq2,mask1,mask2)
        loss = criterion(logits.squeeze(-1), labels.long())
        val_loss+=loss.data.item()
      val_loss= np.mean(val_loss)   
      if(len(val_losses)>0 and val_loss<min(val_losses)):
        torch.save(model.state_dict(), '/home/svu/e0401988/NLP/BERT_NLI/multinli_model.pt')  
      val_losses.append(val_loss)      
    print('training loss:{} validation loss:{}'.format(train_loss,val_loss))

