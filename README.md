# RumourPrediction
Rumour prediction model using PHEME dataset
Paper-https://www.aclweb.org/anthology/P19-1498.pdf

```Requirements```
pip install requirement.py

```Steps to execute```
For BERT,data is already processed into binary format and <br/>
available under data\Encoding folder <br/>
SKP encodings can be downloaded from https://drive.google.com/drive/folders/14h5MCeF09j-d8u3VvzX6j00aEXOrhMXU?usp=sharing <br/>
Save under Encoding/SKP folder

if you wish to encode data again,<br/>
For BERT encodings, download this file - https://drive.google.com/file/d/1K648Cl9j0TplZXiL393h3CHX48ghM2n8/view?usp=sharing into BERT_CONFIG folder <br/>
comment these lines in rumour_prediction.py:<br/>
from skip_thoughts.skip_thoughts import configuration
from skip_thoughts.skip_thoughts import encoder_manager

rumour_prediction.py -mode process -encoding BERT
For SKP encodings, download this folder - https://drive.google.com/drive/folders/1WvYG-d9fPQuapyVRQ3Rb9_OdBiMf_Zsq?usp=sharing into the encoding folder
rumour_prediction.py -mode process -encoding SKP

```args```
-encoding options-'BERT','SKP','EMT','SKPEMT' [Currently BERT and SKP work]<br/>
-tree options-'normal','BCTree' [Currently works for normal]<br/>
-mode options-'train','process' [train trains the model on processed binarized data,<br/>
process processes data to create binarized data] 
-save options-'yes','no' [saves report after training the model <br/>if yes otherwise prints results in console]

```Data``` <br/>
Processed as well as raw data is available in the data/Encoding folder <br/>
It contains both balanced as well as unbalanced dataset. <br/>
Balancing is done by duplicating minority classes trees for each event type <br/>


```Encodings```
* SKP encodings from tensorflow bidirectional implementation of skipthought availabel at <br/>
-https://github.com/elvisyjlin/skip-thoughts.git <br/>
embedding size - 2400
* DeepMoji encoding from https://github.com/zzsza/DeepMoji-Python3.git <br/>
encode_texts.py in examples has been modified to return embeddings <br/>
embedding size-2304
* BERT encoding derived from fine tuning BERT (using transformers library) <br/>
on Multi NLI data available at https://cims.nyu.edu/~sbowman/multinli/ <br/>
embedding size-768
* SKPEMT embedding by concatenating SKP and DeepMoji embeddings

```To do```
* Add options for BC Tree
* Add option for model variations such as child sum, convolve, convolve + concat etc, as mentioned in paper

