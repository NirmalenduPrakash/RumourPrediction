# RumourPrediction
Rumour prediction model using PHEME dataset
Paper-https://www.aclweb.org/anthology/P19-1498.pdf

```Data``` <br/>
Processed as well as raw data is available at <br/> 
https://drive.google.com/drive/folders/14t8CN6HvR1s2kXk03tSdtxxHYJeuo1C0?usp=sharing <br/>
It contains both balanced as well as unbalanced dataset. <br/>
Balancing is done by duplicating minority classes trees for each event type <br/>

* Trained models are available in the models folder
* BERT model is available at https://drive.google.com/file/d/1Dky5KHCTL8r8eudquJopTWglBk05LKf-/view?usp=sharing

```Encodings```
* SKP encodings from tensorflow bidirectional implementation of skipthought availabel at <br/>
-https://github.com/elvisyjlin/skip-thoughts.git <br/>
embedding size - 2400
* DeepMoji encoding from https://github.com/zzsza/DeepMoji-Python3.git <br/>
encode_texts.py in examples has been modified to return embeddings <br/>
embedding size-2304
* BERT encoding derived from fine tuning BERT (using transformers library) <br/>
on Multi NLI data available at https://cims.nyu.edu/~sbowman/multinli/ <br/>
20000 records for training and 5000 for validation <br/>
embedding size-768
* SKPEMT embedding by concatenating SKP and DeepMoji embeddings



