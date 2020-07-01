# %tensorflow_version 1.x
import json
import glob2
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
import regex as re
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
import os
import pickle
import random
from transformers import BertTokenizer   
from transformers import BertModel
import ast
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import argparse
from skip_thoughts.skip_thoughts import configuration
from skip_thoughts.skip_thoughts import encoder_manager
# for DeepMoji Encoder
# import examples.encode_texts

cwd=os.path.dirname(os.path.realpath(__file__))
os.chdir(cwd)
print(cwd)

class Node():
  def __init__(self,x=None,y=None,id=None,children=None):
     self.x=x
     self.id=id
     self.y=y
     self.children=children
     self.h=None
     self.c=None

class Global:
    def __init__(self):
      self.g_nodes=[] 
    def reset(self):
      self.g_nodes=[]

def get_key_value(dicts,search_id,keys=None):
  res={}
  for d in dicts:
    if('id' in d.keys()):
      if(str(d['id'])==str(search_id)):
        for k in keys:
          res[k]=d[k]
        break
    else:
      if(search_id in list(d.keys())):  
        return d[search_id]
      else:
        return None        
  return list(res.values()) 

def uniform_embed(size):
  return np.random.uniform(low=-0.5, high=0.5, size=(size))

def create_tree(tree,tweets=None,stance_dict=None,embed_size=2400):
  global gb
  children=[]
  for key,value in tree.items():
    if(type(value)==dict):      
      child=create_tree(value,tweets,stance_dict,embed_size=embed_size)
      embed=get_key_value(dicts=tweets,search_id=key,keys=['text'])
      if(len(embed)>0 and len(embed[0])>0):
        embed=embed[0]
      else:
        embed=uniform_embed(embed_size) 
      stance=get_key_value(dicts=[stance_dict],search_id=key)        
      parent_node=Node(id=key,x=embed,children=child,y=stance)
      children.append(parent_node)
      gb.g_nodes.append(parent_node)
    else:
      embed=get_key_value(dicts=tweets,search_id=key,keys=['text'])
      if(len(embed)>0 and len(embed[0])>0):
        embed=embed[0]
      else:
        embed=uniform_embed(embed_size) 
      stance=get_key_value(dicts=[stance_dict],search_id=key)           
      child=Node(id=key,x=embed,y=stance)
      children.append(child)
      gb.g_nodes.append(child) 
  return children

class SentenceClassifier(nn.Module):
  def __init__(self):
      super(SentenceClassifier, self).__init__()
      self.bert_layer = BertModel.from_pretrained('BERT_CONFIG')
      self.linear = nn.Linear(768*2, 3)
      self.softmax = nn.Softmax()

  def forward(self,seq,mask):
      _,cls = self.bert_layer(seq, attention_mask = mask)
      return cls.tolist()

"""# Encoders"""
class GLOVE:
  def __init__(self,embedding_file):
    self.vocab={}
    self.unk=[0 for i in range(300)]
    with open(embedding_file, 'rt', encoding='utf-8') as f:
      for line in f:
        splitline=line.split()
        self.vocab[splitline[0]]=[float(value) for value in splitline[1:]]
    self.size=300    
  def embed(self,tokens):
    embed = [self.vocab.get(token,self.unk) for token in tokens]
    embed = [sum(x)/len(x) for x in zip(*embed)]
    return embed

class DeepMoji:
  def __init__(self):
    self.size=2304
  def embed(self,tokens):
    embed=examples.encode_texts.encode(np.array([''.join(tokens)]))
    return embed.reshape(2304)

class SKP:
  def __init__(self,encoder):
    self.encoder=encoder
    self.size=2400
  def embed(self,tokens):
    embed=self.encoder.encode(tokens) 
    embed=[sum(x)/len(x) for x in zip(*embed)]
    return embed

class skpemt:
  def __init__(self,encoder):
    self.encoder=encoder
    self.size=2400+2304
  def embed(self,tokens):
    encoding=self.encoder.encode(tokens) 
    encoding=[sum(x)/len(x) for x in zip(*encoding)]
    deepmoji_encoding=examples.encode_texts.encode(np.array([''.join(tokens)]))
    deepmoji_encoding= list(deepmoji_encoding.reshape(2304))
    encoding.extend(deepmoji_encoding)
    return encoding 

class BERT:
  def __init__(self):
    self.model=SentenceClassifier()
    self.model.load_state_dict(torch.load('BERT_CONFIG/multinli_model.pt'))
    self.size=768
    self.tokenizer = BertTokenizer.from_pretrained('BERT_CONFIG')    
  def tokenize(self,tokens):
    return [self.tokenizer.convert_tokens_to_ids('[CLS]')]\
      +[self.tokenizer.convert_tokens_to_ids(tok) for tok in tokens]
  def embed(self,tokens):
    tokenized=torch.tensor(self.tokenize(tokens)).view(1,-1)
    mask=torch.tensor([1 for i in range(tokenized.size(-1))]).view(1,-1)
    return self.model(tokenized,mask)

"""# Preprocessing Input"""

def create_datapoint(source,reply_list,struct,stance_dict,embedding):
    tweets=reply_list+[source]
    create_tree(struct,tweets,stance_dict,embed_size=embedding.size)

def preprocess(tweet):
    # remove @mentions, RT,MT,DM,PRT,HT,CC, URLs
    contractions = { 
      "n't": "not",
      "'ve": "have",
      "'d": "would",
      "'ll": "will",
      "'m": "am",
      "ma'am": "madam",
      "'re": "they are"
      }
    text=tweet.lower()  
    tokens=text.split()
    # print(tokens)
    for indx,token in enumerate(tokens):
        for contra in contractions.keys():
            if(contra in token):
                tokens[indx]=' '.join([token.replace(contra,''),contractions[contra]])
    text=' '.join(tokens)
    # Format words and remove unwanted characters    
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'([@?])(\w+)\b', ' ', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    text=text.replace('mt ',' ').replace('rt ',' ').replace('dm ',' ').replace('prt ',' ').replace('ht ',' ').replace('cc ',' ')
    tokens = text.split()
    # stops = set(stopwords.words("english"))
    # tokens = [w for w in tokens if not w in stops]
    return tokens

"""# Model"""
class TreeLSTMCell(nn.Module):
  def __init__(self,x_size,h_size):
    super(TreeLSTMCell, self).__init__()
    self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
    self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
    self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))
    self.U_f = nn.Linear(h_size, h_size)
    self.b_f = nn.Parameter(torch.zeros(1, h_size))      
    self.h=torch.zeros(1,h_size)
    self.c=torch.zeros(1,h_size)
    self.sigmoid=nn.Sigmoid()
    self.tanh=nn.Tanh()
    self.conv=nn.Conv1d(in_channels=1,out_channels=1, kernel_size=2, stride=1,padding=1)
    self.maxpool=nn.AdaptiveMaxPool1d(h_size)
    self.stance_linear=nn.Linear(h_size,4)
    self.veracity_linear=nn.Linear(h_size,3)    
    self.h_size=h_size
  def reset_h_c(self):
    self.h=torch.zeros(1,self.h_size)
    self.c=torch.zeros(1,self.h_size)    
  def forward(self,node,is_last=False):      
    if(node.children is None):
      x=torch.tensor(node.x,dtype=torch.float)
      x_iou=self.W_iou(x) 
      u_iou=self.U_iou(self.h)
      b_iou=self.b_iou      
      xs=torch.chunk(x_iou,3,-1)
      us=torch.chunk(u_iou,3,-1)
      bs=torch.chunk(b_iou,3,-1)       
      i=self.sigmoid(xs[0]+us[0]+bs[0])
      o=self.sigmoid(xs[1]+us[1]+bs[1])
      u=self.sigmoid(xs[2]+us[2]+bs[2])
      self.c=i*u
      self.h=o*self.tanh(self.c)
      node.h=self.h
      node.c=self.c
      if(is_last):
        veracity=self.veracity_linear(self.h)
        return veracity
      else:
        stance=self.stance_linear(self.h)
        return stance           
    else:
      h_stack=torch.stack([child.h for child in node.children])
      u_f=self.U_f(h_stack.view(-1,self.h_size))
      f=self.sigmoid(u_f+self.b_f)
      h_hat,_=torch.max(self.maxpool(self.conv(h_stack)),dim=0,keepdim=True)
      u_iou=self.U_iou(h_hat.view(1,-1))

      us=torch.chunk(u_iou,3,-1)
      bs=torch.chunk(self.b_iou,3,-1)            
      i=self.sigmoid(us[0]+bs[0])
      o=self.sigmoid(us[1]+bs[1])
      u=self.sigmoid(us[2]+bs[2])
      self.c=i*u + torch.sum(f.view(-1,self.h_size)*torch.stack([child.c for child in node.children]).view(-1,self.h_size),0)
      self.h=o*self.tanh(self.c)
      node.h=self.h
      node.c=self.c
      if(is_last):
        veracity=self.veracity_linear(self.h)
        return veracity 
      stance=self.stance_linear(self.h)
      return stance

"""# Loading Data"""
veracity_enum = { 
  'true':0,
  'false':1,
  'unverified':2
}

stance_enum={
    'agreed':0,
    'comment':1,
    'appeal-for-more-information':2,
    'disagreed':3
}

class Event:
  def __init__(self,path):
    self.path=path
    self.treelist=[]
  def create_tree(self,gb):
    for r in ['rumours']:
      dirs=os.listdir(self.path + '/' + r)
      for d in dirs:
        d=d.replace('._','')
        if(d=='.DS_Store'):
          continue
        source=glob2.glob(self.path+'/'+r+'/'+ d + '/source-tweets/*.json')
        if(source is not None and len(source)>0):
          source=source[0]
        else:
          print('source tweet not available for {}'.format(d))  
          continue
        replies=glob2.glob(self.path+'/'+r+'/'+ d + '/reactions/*.json')
        structure=self.path+'/'+ r + '/' + d + '/structure.json'
        gb.reset()
        reply_list=[]
        with open(source,'r') as f:
          source=json.load(f)
        processed_text= source['text'].split()
        if(len(processed_text)>0):  
          source={'id':source['id'],'text':embedding.embed(processed_text)}  
        else:
          source={'id':source['id'],'text':processed_text}        
        for reply in replies:
          with open(reply,'r') as f:
            reply=json.load(f)
          processed_text= reply['text'].split()
          if(len(processed_text)>0):
            reply={'id':reply['id'],'text':embedding.embed(processed_text)}
          else:
            reply={'id':reply['id'],'text':processed_text} 
          reply_list.append(reply)
          # print(reply_list)      
        with open(structure,'r') as f:
          struct=json.load(f)
        create_datapoint(source,reply_list,struct,stance_dict,embedding=embedding)
        self.treelist.append((source['id'],gb.g_nodes))

def convert_annotations(annotation, string = True):
    if 'misinformation' in annotation.keys() and 'true'in annotation.keys():
        if int(annotation['misinformation'])==0 and int(annotation['true'])==0:
            if string:
                label = "unverified"
            else:
                label = 2
        elif int(annotation['misinformation'])==0 and int(annotation['true'])==1 :
            if string:
                label = "true"
            else:
                label = 1
        elif int(annotation['misinformation'])==1 and int(annotation['true'])==0 :
            if string:
                label = "false"
            else:
                label = 0
        elif int(annotation['misinformation'])==1 and int(annotation['true'])==1:
            print ("OMG! They both are 1!")
            print(annotation['misinformation'])
            print(annotation['true'])
            label = None
            
    elif 'misinformation' in annotation.keys() and 'true' not in annotation.keys():
        # all instances have misinfo label but don't have true label
        if int(annotation['misinformation'])==0:
            if string:
                label = "unverified"
            else:
                label = 2
        elif int(annotation['misinformation'])==1:
            if string:
                label = "false"
            else:
                label = 0
                
    elif 'true' in annotation.keys() and 'misinformation' not in annotation.keys():
        print ('Has true not misinformation')
        label = None
    else:
        print('No annotations')
        label = None
           
    return label

def load_rumour_veracity_labels():
  root_dir='all-rnr-annotated-threads'
  for e in event_labels:
    for r in ['rumours']:
      dirs=os.listdir(root_dir  + '/{}'.format(e) + '/' + r)
      for d in dirs:
        d=d.replace('._','')
        if(d=='.DS_Store'):
          continue
        with open(root_dir+'/'+e+'/'+r+'/'+d+ '/annotation.json','r') as f:
          veracity=json.load(f)
          if(veracity is not None):
            veracity_dict.update({d:veracity_enum[convert_annotations(veracity)]})
          else:
            print('annotation none for :{}'.format(d))       

# # load stance labels
def load_stance_labels():
  stance_annotation_file='stance-annotations.json'
  with open(stance_annotation_file,'r') as f:
    lines=f.readlines()
  for line in lines:
    if('#' in line or 'putinmissing' in line \
      or 'prince-toronto' in line or 'ebola-essien' in line):
      continue
    annot=ast.literal_eval(line)
    if('responsetype-vs-previous' in annot.keys()):
      stance_dict[annot['tweetid']]=stance_enum[annot['responsetype-vs-previous']]
    elif('responsetype-vs-source' in annot.keys()):
      stance_dict[annot['tweetid']]=stance_enum[annot['responsetype-vs-source']]
 
def load_labels():
  if(os.path.isfile('veracity_dict.pkl')):
    with open('veracity_dict.pkl','rb') as f:
      veracity_dict=pickle.load(f) 
  else:
    load_rumour_veracity_labels() 

  if(os.path.isfile('veracity_dict.pkl')):
    with open('stance_dict.pkl','rb') as f:
      stance_dict=pickle.load(f)  
  else:
    load_stance_labels()            

def dump_embedding(typ):
  root_path='all-rnr-annotated-threads/'
  for e in event_labels#['charliehebdo','sydneysiege','germanwings-crash','ferguson','ottawashooting']:
    ev=Event(root_path+e)
    ev.create_tree(gb)
    with open('data/Encoding/{}/data_{}{}.pkl'.format(typ,tree_lbl,e),'wb') as f:
      pickle.dump(ev,f)

# embedding=BERT()
# # embedding=DeepMoji()
# # embedding=SKP(encoder)
# gb=Global()
# # dump_embedding('SKP')
# dump_embedding('BERT')

"""# Verify Data Distribution"""
# ev_label_count=[]
# for e in ['charliehebdo','sydneysiege','germanwings-crash','ferguson','ottawashooting']:
#   with open('/content/drive/My Drive/PHEME/Encoding/BERT/data_UnlimitedTree_{}.pkl'.format(e),'rb') as f:
#     ev=pickle.load(f)
#   label={0:0,1:0,2:0,3:0}   
#   # for event in ev:
#   for tree in ev.treelist:
#     nodes=tree[1]
#     for node in nodes:
#       nodeid_list.append(node.id)
#       if(node.y==0):
#         count+=1
#       if(node.y is not None):
#         label[node.y]+=1    
#   ev_label_count.append((e,label))    
# print(ev_label_count)

# ev_label_count=[]
# for e in ['charliehebdo','sydneysiege','germanwings-crash','ferguson','ottawashooting']:
#   with open('/content/drive/My Drive/PHEME/Encoding/BERT/data_UnlimitedTree_{}.pkl'.format(e),'rb') as f:
#     ev=pickle.load(f)
#   label={0:0,1:0,2:0}   
#   # for event in ev:
#   for tree in ev.treelist:
#     label[veracity_dict[str(tree[0])]]+=1
#   ev_label_count.append((e,label))    
# print(ev_label_count)

# Balancing each class for rumour labels
def balance_embedding(typ):
  ev_label_count=[]
  for e in event_labels#['charliehebdo','sydneysiege','germanwings-crash','ferguson','ottawashooting']:
    with open('data/Encoding/{}/data_{}{}.pkl'.format(typ,tree_lbl,e),'rb') as f:
      ev=pickle.load(f)
    label={0:0,1:0,2:0}   
    # for event in ev:
    for tree in ev.treelist:
      label[veracity_dict[str(tree[0])]]+=1
    ev_label_count.append((e,label))    
  print(ev_label_count)
  for e in event_labels#['charliehebdo','sydneysiege','germanwings-crash','ferguson','ottawashooting']:
    with open('data/Encoding/{}/data_{}{}.pkl'.format(typ,tree_lbl,e),'rb') as f:
      ev=pickle.load(f) 
    # for event in ev:
    lbl_0_list=[]
    lbl_1_list=[]
    lbl_2_list=[]
    for tree in ev.treelist: 
      if(veracity_dict[str(tree[0])]==0):
        lbl_0_list.append(tree)      
      elif(veracity_dict[str(tree[0])]==1):
        lbl_1_list.append(tree)
      elif(veracity_dict[str(tree[0])]==2):
        lbl_2_list.append(tree)
    if(e=='charliehebdo'):
      for i in range(77):
        ev.treelist.append(random.sample(lbl_1_list,1)[0])
      for i in range(44):
        ev.treelist.append(random.sample(lbl_2_list,1)[0])
    elif(e=='sydneysiege'):
      for i in range(296):
        ev.treelist.append(random.sample(lbl_1_list,1)[0])
      for i in range(328):
        ev.treelist.append(random.sample(lbl_2_list,1)[0])   
    elif(e=='germanwings-crash'):
      for i in range(77):
        ev.treelist.append(random.sample(lbl_2_list,1)[0]) 
    elif(e=='ferguson'):
      for i in range(256):
        ev.treelist.append(random.sample(lbl_0_list,1)[0])
      for i in range(258):
        ev.treelist.append(random.sample(lbl_1_list,1)[0])        
    elif(e=='ottawashooting'):
      for i in range(257):
        ev.treelist.append(random.sample(lbl_1_list,1)[0])    
      for i in range(260):
        ev.treelist.append(random.sample(lbl_2_list,1)[0])    
    with open('data/Encoding/{}/data_{}{}_balanced.pkl'.format(typ,tree_lbl,e),'wb') as f:
      pickle.dump(ev,f) 

  ev_label_count=[]
  for e in event_labels#['charliehebdo','sydneysiege','germanwings-crash','ferguson','ottawashooting']:
    with open('data/Encoding/{}/data_{}{}_balanced.pkl'.format(typ,tree_lbl,e),'rb') as f:
      ev=pickle.load(f)
    label={0:0,1:0,2:0}   
    # for event in ev:
    for tree in ev.treelist:
      label[veracity_dict[str(tree[0])]]+=1
    ev_label_count.append((e,label))    
  print(ev_label_count) 


"""# Training"""
def train_loader(val_index):
  for e in [l for i,l in enumerate(event_labels) if i!=val_index]:
    with open('data/Encoding/{}/data_{}{}_balanced.pkl'\
          .format(encoding,tree_lbl,e),'rb') as f:
      ev=pickle.load(f)
    yield ev,e

def val_loader(val_index):
  # for e in [l for i,l in enumerate(event_labels) if i==val_index]:
  with open('data/Encoding/{}/data_{}{}.pkl'\
            .format(encoding,tree_lbl,event_labels[val_index]),'rb') as f:
    ev=pickle.load(f)
  return ev,event_labels[val_index]


def measure(val_index):
    with torch.no_grad():
      # for val_index in range(5):
      # for event,lbl in val_loader(val_index):
      event,lbl=val_loader(val_index)
      # print('testing on {}...'.format(lbl))    
      for tree in event:
        nodes=tree[1]
        for i in range(len(nodes)):
          nodes[i].h=None
          nodes[i].c=None
        model.reset_h_c()  
        for i in range(len(nodes)):
          if(i!=(len(nodes)-1)):
            result=model(nodes[i])
            if(result is not None and nodes[i].y is not None):
              results['stance'][lbl].append((nodes[i].y,torch.argmax(result,dim=-1).item()))              
          else:
            result=model(nodes[i],is_last=True) 
            results['veracity'][lbl].append((veracity_dict[str(tree[0])],torch.argmax(result,dim=-1).item()))

def calculate_val_loss(index):
  with torch.no_grad():
    loss=0
    stance_loss=0 
    event,lbl=val_loader(index) 
    print('testing on {}...'.format(lbl))
    for tree in event:
      nodes=tree[1]
      for i in range(len(nodes)):
        nodes[i].h=None
        nodes[i].c=None
      model.reset_h_c()  
      for i in range(len(nodes)):
        if(i!=(len(nodes)-1)):
          result=model(nodes[i])
          if(nodes[i].y is not None):
            stance_loss+=criterion(result,torch.tensor(nodes[i].y).view(-1))            
        else:
          result=model(nodes[i],is_last=True)      
          loss+=criterion(result,torch.tensor(veracity_dict[str(tree[0])],dtype=torch.long).view(-1))  
    print('veracity loss :{} stance loss :{}'.format(loss.data.item(),stance_loss.data.item()))             
    return loss.data.item() + stance_loss.data.item()

# Training
def train():  
  for e in range(5):  
    print('==================== Iteration {} =========='.format(e))
    val_index=e%5 
    model=TreeLSTMCell(768,768)
    optimizer = optim.Adam(model.parameters(), lr=.008)
    train_loss=0
    val_losses=[]
    for iter in range(30):
      for event,lbl in train_loader(val_index):
        print('training on {}...'.format(lbl))    
        for indx,tree in enumerate(event):
          loss=0
          stance_loss=0
          optimizer.zero_grad()  
          nodes=tree[1]
          for i in range(len(nodes)):
            nodes[i].h=None
            nodes[i].c=None
          model.reset_h_c()  
          stance_count=0
          for i in range(len(nodes)):
            if(i!=(len(nodes)-1)):
              result=model(nodes[i])
              # for few records annotation not available
              if(nodes[i].y is not None):
                stance_loss+=criterion(result,torch.tensor(nodes[i].y).view(-1))
                stance_count+=1
            else:
              result=model(nodes[i],is_last=True)                
              loss+=criterion(result,torch.tensor(veracity_dict[str(tree[0])],dtype=torch.long).view(-1))         
          if(iter%2==0 and stance_loss>0):
            stance_loss.backward()
            optimizer.step()
          elif(iter%2!=0):
            loss.backward() 
            optimizer.step() 
          loss+=stance_loss 
          train_loss+=loss.data.item()                
        print('train loss : {}'.format(train_loss))
      val_loss= calculate_val_loss(val_index)
      if (iter==0):
        measure(val_index)  
      elif (len(val_losses)>0 and val_loss<min(val_losses)):
        measure(val_index)    
      val_losses.append(val_loss) 

def report():
  if(args.save=='yes'):
    f=open('report.txt','w+')
  """# Metrics"""
  for lbl in event_labels#['charliehebdo','sydneysiege','germanwings-crash','ferguson','ottawashooting']:
    y_true=[y[0] for y in results['veracity'][lbl]]
    y_pred=[y[1] for y in results['veracity'][lbl]]

    if(args.save=='yes'):
      f.write('=============={}==Veracity Report=============='.format(lbl))
      f.write(classification_report(y_true,y_pred))
    else:  
      print('=============={}==Veracity Report=============='.format(lbl))
      print(classification_report(y_true,y_pred))

    y_true=[y[0] for y in results['stance'][lbl]]
    y_pred=[y[1] for y in results['stance'][lbl]]
    if(args.save=='yes'):
      f.write('=============={}==Stance Report=============='.format(lbl))
      f.write(classification_report(y_true,y_pred))  
    else:      
      print('=============={}==Stance Report=============='.format(lbl))
      print(classification_report(y_true,y_pred))

  # Overall veracity confusion matrix
  y_true=[]
  y_pred=[]
  for lbl in event_labels#['ottawashooting','sydneysiege','germanwings-crash','ferguson','ottawashooting']:
    y_true.extend([y[0] for y in results['veracity'][lbl]])
    y_pred.extend([y[1] for y in results['veracity'][lbl]])
  if(args.save=='yes'):
      f.write('==============Veracity Confucion matrix=============='.format(lbl))
      f.write(confusion_matrix(y_true,y_pred,normalize='true'))
      f.close()  
  else:          
    print('================Veracity Confucion matrix==============')
    print(confusion_matrix(y_true,y_pred,normalize='true'))

if  __name__== "__main__":
  results={'stance':{'charliehebdo':[],'sydneysiege':[],\
                    'germanwings-crash':[],'ferguson':[],'ottawashooting':[]}, \
                    'veracity':{'charliehebdo':[],'sydneysiege':[],\
                    'germanwings-crash':[],'ferguson':[],'ottawashooting':[]}}
  veracity_dict={}
  stance_dict={}
  model=None
  criterion = nn.CrossEntropyLoss()
  event_labels=['charliehebdo','sydneysiege','germanwings-crash','ferguson','ottawashooting']

  parser = argparse.ArgumentParser()
  parser.add_argument("-encoding", default='BERT', type=str, choices=['BERT','SKP','EMT','SKPEMT'])
  parser.add_argument("-tree", default='Normal', type=str, choices=['normal','BCTree'])
  parser.add_argument("-mode", default='train',type=str,choices=['train','process'])
  parser.add_argument("-save", default='yes',type=str,choices=['yes','no'])
  
  args = parser.parse_args()
  encoding=args.encoding  
  mode=args.mode

  if(args.tree=='normal'):
    tree_lbl='UnlimitedTree_'
  else:
    tree_lbl=''  

  if(mode=='train'):
    train()
    report()
  elif(mode=='process'):
    load_labels()
    if(args.encoding=='SKP'):
      VOCAB_FILE = "skip_thoughts/skip_thoughts_bi_2017_02_16/vocab.txt"
      EMBEDDING_MATRIX_FILE = "skip_thoughts/skip_thoughts_bi_2017_02_16/embeddings.npy"
      CHECKPOINT_PATH = "skip_thoughts/skip_thoughts_bi_2017_02_16/model.ckpt-500008"

      encoder = encoder_manager.EncoderManager()
      encoder.load_model(configuration.model_config(bidirectional_encoder=True),
                        vocabulary_file=VOCAB_FILE,
                        embedding_matrix_file=EMBEDDING_MATRIX_FILE,
                        checkpoint_path=CHECKPOINT_PATH)
      embedding=SKP(encoder)
    elif(args.encoding=='BERT'):
      embedding=BERT()  
    dump_embedding(args.embedding)
    balance_embedding(args.embedding)



  