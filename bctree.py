import json
import glob2
import nltk
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
import examples.encode_texts

class Node():
  def __init__(self,x=None,y=None,id=None,node1=None,node2=None):
    global gb
    if(id is None and (node1 is None or node2 is None)):
      raise Exception("To create non virtual nodes, id is required")
    if(id is not None):
      self.id=str(id)
      self.is_leaf_node=True
    else:
      gb.varint+=1
      self.id='vir_'+ str(gb.varint)
      self.node1=node1
      self.node2=node2  
      if('vir_' in node1.id and 'vir_' in node2.id):
        self.is_stance_node=False
      else:
        self.is_stance_node=True
      self.is_leaf_node=False  
    self.x=x
    self.y=y
    self.c=None
    self.h=None

class Global:
    def __init__(self):
      self.varint=0
      self.g_parent=None
      self.g_nodes=[] 
    def reset(self):
      self.varint=0
      self.g_parent=None
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
      if(search_id in d.keys()):  
        return d[search_id]
      else:
        return None        
  return list(res.values()) 

def uniform_embed(size):
  return np.random.uniform(low=-0.5, high=0.5, size=(size))

def create_tree(tree,tweets=None,stance_dict=None,parent=None,embed_size=2400):
  global gb
  children=[]
  for key,value in tree.items():
    if(type(value)==dict):      
      child=create_tree(value,tweets,stance_dict,parent=key,embed_size=embed_size)
    else:
      embed=get_key_value(dicts=tweets,search_id=key,keys=['text'])
      if(len(embed)>0 and len(embed[0])>0):
        embed=embed[0]
      else:
        embed=uniform_embed(embed_size)            
      child=Node(id=key,x=embed)
    children.append(child) 
  if(gb.g_parent is not None):
    arr=[gb.g_parent]
    arr.extend(children)
    children=arr 
  if(parent is not None):
    nodes=[]
    for child in children:
      if('vir_' not in str(parent) and 'vir_' in child.id):
        nodes.append(child)
        continue
      embed=get_key_value(dicts=tweets,search_id=parent,keys=['text']) 
      if(len(embed)>0 and len(embed[0])>0):
        embed=embed[0]  
      else:
        embed=uniform_embed(embed_size)      
      parent_node=Node(id=parent,x=embed) 
      stance=None
      if('var_' not in parent_node.id and 'var_' not in child.id):
        stance=get_key_value(dicts=[stance_dict],search_id=child.id) 
      node=Node(node1=parent_node,node2=child,y=stance)
      # print(parent,child.id,node.id)
      gb.g_nodes.extend([parent_node,child,node])
      nodes.append(node)
    node1=nodes[0] 
    for node_iter in range(1,len(nodes)):
      node2=nodes[node_iter]
      node=Node(node1=node1,node2=node2)
      # print(node1.id,node2.id,node.id)
      gb.g_nodes.extend([node])
      node1=node 
    embed=get_key_value(dicts=tweets,search_id=parent,keys=['text']) 
    if(len(embed)>0 and len(embed[0])>0):
      embed=embed[0]  
    else:
      embed=uniform_embed(embed_size)            
    gb.g_parent=Node(id=parent,x=embed)
    return node1
  return children

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
    stops = set(stopwords.words("english"))
    tokens = [w for w in tokens if not w in stops]
    return tokens

class TreeLSTMCell(nn.Module):
  def __init__(self,x_size,h_size):
    super(TreeLSTMCell, self).__init__()
    self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
    self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
    self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))
    self.U_f = nn.Linear(2 * h_size, 2 * h_size)
    self.b_f = nn.Parameter(torch.zeros(1, 2 * h_size))      
    self.h=torch.zeros(1,h_size)
    self.c=torch.zeros(1,h_size)
    self.sigmoid=nn.Sigmoid()
    self.tanh=nn.Tanh()
    self.conv=nn.Conv2d(in_channels=1,out_channels=1, kernel_size=2, stride=1,padding=(0,1))
    self.maxpool=nn.MaxPool2d(kernel_size=(1,2),stride=1)
    self.stance_linear=nn.Linear(h_size,4)
    self.veracity_linear=nn.Linear(h_size,3)
    self.softmax=nn.Softmax()
    self.h_size=h_size
  def reset_h_c(self):
    self.h=torch.zeros(1,self.h_size)
    self.c=torch.zeros(1,self.h_size)    
  def forward(self,node,is_last=False):      
    if(node.is_leaf_node):
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
    else:
      h_stack=torch.stack([node.node1.h,node.node2.h])
      u_f=self.U_f(h_stack.view(1,-1))
      f=self.sigmoid(u_f+self.b_f)
      h_hat=self.maxpool(self.conv(h_stack.permute(1,0,2).unsqueeze(0))).view(1,-1)
      u_iou=self.U_iou(h_hat)
      us=torch.chunk(u_iou,3,-1)
      bs=torch.chunk(self.b_iou,3,-1)            
      i=self.sigmoid(us[0]+bs[0])
      o=self.sigmoid(us[1]+bs[1])
      u=self.sigmoid(us[2]+bs[2])
      self.c=i*u + torch.sum(f.view(-1,self.h_size)*torch.stack([node.node1.c,node.node2.c]).view(-1,self.h_size),0)
      self.h=o*self.tanh(self.c)
      node.h=self.h
      node.c=self.c
      if(node.is_stance_node):
        stance=self.softmax(self.stance_linear(self.h))
        return stance        
      if(is_last):
        veracity=self.softmax(self.veracity_linear(self.h))
        return veracity 

"""# Loading Data"""

veracity_enum = { 
  'true':0,
  'false':1,
  'unverified':2
}

stance_enum={
    'support':0,
    'comment':1,
    'query':2,
    'deny':3
}

class Event:
  def __init__(self,path):
    self.path=path
    self.treelist=[]
  def create_tree(self,gb):
    dirs=os.listdir(self.path)
    for d in dirs:
      if(d=='.DS_Store'):
        continue
      source=glob2.glob(self.path+'/'+ d + '/source-tweet/*.json')[0]
      replies=glob2.glob(self.path+'/'+ d + '/replies/*.json')
      structure=self.path+'/'+ d + '/structure.json'
      gb.reset()
      reply_list=[]
      with open(source,'r') as f:
        source=json.load(f,parse_int=None)
      processed_text= preprocess(source['text'])  
      if(len(processed_text)>0):  
        source={'id':source['id'],'text':embedding.embed(processed_text)}  
      else:
        source={'id':source['id'],'text':processed_text}        
      for reply in replies:
        with open(reply,'r') as f:
          reply=json.load(f,parse_int=None)
        processed_text= preprocess(reply['text'])
        if(len(processed_text)>0):
          reply={'id':reply['id'],'text':embedding.embed(processed_text)}
        else:
          reply={'id':reply['id'],'text':processed_text} 
        reply_list.append(reply)      
      with open(structure,'r') as f:
        struct=json.load(f,parse_int=None)
      create_datapoint(source,reply_list,struct,stance_dict,embedding=embedding)
      self.treelist.append((source['id'],gb.g_nodes))

def dump_embedding(typ):
  root_path = dirname(dirname(abspath(__file__))) 
  for e in ['charliehebdo','sydneysiege','germanwings-crash','ferguson','ottawashooting']:
    # event_list=[]
    ev=Event(root_path+e)
    ev.create_tree(gb)
    # event_list.append(ev)
    with open('{}/data/DeepMoji/data_{}_{}.pkl'.format(root_path,typ,e),'wb') as f:
      pickle.dump(ev,f)

# with open('/content/drive/My Drive/data.pkl','rb') as f:
#   event_list=pickle.load(f)

# Balancing each class for rumour labels
def balance_embedding(typ):
  root_path = dirname(dirname(abspath(__file__))) 
  ev_label_count=[]
  for e in ['charliehebdo','sydneysiege','germanwings-crash','ferguson','ottawashooting']:
    with open('{}/data/DeepMoji/data_{}_{}.pkl'.format(root_path,typ,e),'rb') as f:
      ev=pickle.load(f)
    label={0:0,1:0,2:0}   
    # for event in ev:
    for tree in ev.treelist:
      label[veracity_dict[str(tree[0])]]+=1
    ev_label_count.append((e,label))    
  print(ev_label_count)
  for e in ['charliehebdo','sydneysiege','germanwings-crash','ferguson','ottawashooting']:
    with open('{}/data/DeepMoji/data_{}_{}.pkl'.format(root_path,typ,e),'rb') as f:
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
      for i in range(27):
        ev.treelist.append(random.sample(lbl_1_list,1)[0])
      for i in range(16):
        ev.treelist.append(random.sample(lbl_2_list,1)[0])
    elif(e=='sydneysiege'):
      for i in range(38):
        ev.treelist.append(random.sample(lbl_1_list,1)[0])
      for i in range(44):
        ev.treelist.append(random.sample(lbl_2_list,1)[0])   
    elif(e=='germanwings-crash'):
      for i in range(9):
        ev.treelist.append(random.sample(lbl_2_list,1)[0]) 
    elif(e=='ferguson'):
      for i in range(42):
        ev.treelist.append(random.sample(lbl_0_list,1)[0])
    elif(e=='ottawashooting'):
      for i in range(40):
        ev.treelist.append(random.sample(lbl_1_list,1)[0])    
      for i in range(23):
        ev.treelist.append(random.sample(lbl_2_list,1)[0])    
    with open('{}/data/DeepMoji/data_{}_{}_balanced.pkl'.format(root_path,typ,e),'wb') as f:
      pickle.dump(ev,f) 

  ev_label_count=[]
  for e in ['charliehebdo','sydneysiege','germanwings-crash','ferguson','ottawashooting']:
    with open('{}/data/DeepMoji/data_{}_{}_balanced.pkl'.format(root_path,typ,e),'rb') as f:
      ev=pickle.load(f)
    label={0:0,1:0,2:0}   
    # for event in ev:
    for tree in ev.treelist:
      label[veracity_dict[str(tree[0])]]+=1
    ev_label_count.append((e,label))    
  print(ev_label_count) 

event_labels=['charliehebdo','sydneysiege','germanwings-crash','ferguson','ottawashooting']
# change type for embedding skp,deepmoji,bert,mix(deepmoji+skp)
def train_loader(val_index,embed_typ,data_root):
  for e in [l for i,l in enumerate(event_labels) if i!=val_index]:
    with open('{}/data_{}_{}_balanced.pkl'.format(data_root,embed_typ,e),'rb') as f:
      ev=pickle.load(f)
    yield ev,e

def val_loader(val_index,embed_typ,data_root):
  for e in [l for i,l in enumerate(event_labels) if i==val_index]:
    with open('{}/data_{}_{}_balanced.pkl'.format(data_root,embed_typ,e),'rb') as f:
      ev=pickle.load(f)
    yield ev,e

def process_data(embed_typ):
  root_path = dirname(dirname(abspath(__file__))) 
  # load stance labels
  rumour_files=glob2.glob('{}/data/rumoureval-data/traindev/*subtaskA*.json'.format(root_path))
  stance_dict={}
  for file in rumour_files:
    with open(file,'r') as f:
      stance_dict.update(json.load(f,parse_int=None))
  for k,v in stance_dict.items():
    stance_dict[k]=stance_enum[v]

  #load veracity labels
  rumour_files=glob2.glob('{}/data/rumoureval-data/traindev/*subtaskB*.json'.format(root_path))
  veracity_dict={}
  for file in rumour_files:
    with open(file,'r') as f:
      veracity_dict.update(json.load(f,parse_int=None))
  for k,v in veracity_dict.items():
    veracity_dict[k]=veracity_enum[v]

  gb=Global()
  if(embed_typ=='skp'):
    embedding=SKP(encoder)
    dump_embedding('skp')
    balance_embedding('skp')    
  elif(embed_typ=='deepmoji')  
    embedding=DeepMoji()  
    dump_embedding('deepmoji')                                                                    
    balance_embedding('deepmoji')

def train(embed_typ,output_path=None):
  root_path = dirname(dirname(abspath(__file__)))
  if(output_path is None):  
    output_path = '{}/model/model_{}.pt'.format(root_path,embed_typ)
  data_root='{}/data/{}'.format(root_path,embed_typ)

  if(embed_typ=='deepmoji'):
    embed_dim=2304
  elif(embed_typ=='skp'):
    embed_dim=2400  
  model=TreeLSTMCell(embed_dim,embed_dim)
  optimizer = optim.Adam(model.parameters(), lr=.008)
  criterion = nn.CrossEntropyLoss()

  for e in range(30):
    print('============ Iteration {} =========================='.format(e))
    val_index=e%5
    train_loss=0
    for event,lbl in train_loader(val_index,embed_typ,data_root):
      print('training on {}...'.format(lbl))
      for tree in event.treelist:
          loss=0
          optimizer.zero_grad()
          nodes=tree[1]
          for i in range(len(nodes)):
            nodes[i].h=None
            nodes[i].c=None
          model.reset_h_c()  
          for i in range(len(nodes)):
            if(i!=(len(nodes)-1)):
              result=model(nodes[i])
              if(result is not None and nodes[i].y is not None):
                loss+=criterion(result,torch.tensor(nodes[i].y).view(-1))
            else:
              result=model(nodes[i],is_last=True)          
              loss+=criterion(result,torch.tensor(veracity_dict[str(tree[0])],dtype=torch.long).view(-1))            
          train_loss+=loss.data.item()        
          loss.backward()
          optimizer.step()
    print('train loss : {}'.format(train_loss))     
    with torch.no_grad():
      loss=0
      for event,lbl in val_loader(val_index,embed_typ,data_root):
        print('testing on {}...'.format(lbl))
        for tree in event.treelist:
          nodes=tree[1]
          for i in range(len(nodes)):
            nodes[i].h=None
            nodes[i].c=None
          model.reset_h_c()  
          for i in range(len(nodes)):
            if(i!=(len(nodes)-1)):
              result=model(nodes[i])
              if(result is not None and nodes[i].y is not None):
                loss+=criterion(result,torch.tensor(nodes[i].y).view(-1)).data.item()
            else:
              result=model(nodes[i],is_last=True)
              loss+=criterion(result,torch.tensor(veracity_dict[str(tree[0])],dtype=torch.long).view(-1)).data.item() 
      print('test loss : {}'.format(loss))
      torch.save(model.state_dict(), output_path)

"""# Metrics"""

# model=TreeLSTMCell(2400,2400)
# model.load_state_dict(torch.load('/content/drive/My Drive/model.pt'))
# results={}
# for val_index in range(5):
#   with torch.no_grad():
#     for event,lbl in val_loader(val_index):
#       pred=[]
#       print('testing on {}...'.format(lbl))
#       for tree in event[0].treelist:
#         nodes=tree[1]
#         for i in range(len(nodes)):
#           nodes[i].h=None
#           nodes[i].c=None
#         model.reset_h_c()  
#         for i in range(len(nodes)):
#           if(i!=(len(nodes)-1)):
#             result=model(nodes[i])
#           else:
#             result=model(nodes[i],is_last=True)
#             print(result)
#             pred.append((torch.argmax(result).item(),veracity_dict[str(tree[0])]))
#       results[lbl]=pred

