# RumourPrediction
Rumour prediction model using PHEME dataset
Paper-https://www.aclweb.org/anthology/P19-1498.pdf

```Data
Processed data is available at https://drive.google.com/drive/folders/1Fd1UZ8JMJKEgkX9j6xWmwSiulwIT2ux2?usp=sharing 
It contains both balanced as well as unbalanced dataset.
Balancing is done by duplicating minority classes trees for each event type
```

```Encodings
* SKP encodings from tensorflow bidirectional implementation of skipthought availabel at 
-https://github.com/elvisyjlin/skip-thoughts.git
embedding size - 2400
* DeepMoji encoding from https://github.com/zzsza/DeepMoji-Python3.git
encode_texts.py in examples has been modified to return embeddings
embedding size-2304
* BERT encoding derived from fine tuning BERT (using transformers library) on Multi NLI data available at https://cims.nyu.edu/~sbowman/multinli/
20000 records for training and 5000 for validation
embedding size-768
* SKPEMT embedding by concatenating SKP and DeepMoji embeddings
```


