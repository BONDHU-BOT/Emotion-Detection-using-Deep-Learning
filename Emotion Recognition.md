# Contextual Emotion Detection


## 1. Loading Data


```python
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
import nltk
import re
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, GRU, LSTM, Bidirectional, Embedding, Dropout
from keras.callbacks import ModelCheckpoint
```

Dataset link: 


```python
def load_dataset(filename):
  df = pd.read_csv(filename)
  label = df["label"]
  unique_label = list(set(label))
  sentences = list(df["text"])
  
  return (df, label, unique_label, sentences)
```


```python
df, label, unique_label, sentences = load_dataset('iseardataset.csv')
```


```python
print(unique_label)
```

    ['disgust', 'fear', 'anger', 'joy', 'guilt', 'shame', 'sadness']



```python
print(df.head(10))
```

         label                                               text Unnamed: 2
    0      joy  On days when I feel close to my partner and ot...        NaN
    1     fear  Every time I imagine that someone I love or I ...        NaN
    2    anger  When I had been obviously unjustly treated and...        NaN
    3  sadness  When I think about the short time that we live...        NaN
    4  disgust  At a gathering I found myself involuntarily si...        NaN
    5    shame  When I realized that I was directing the feeli...        NaN
    6    guilt  I feel guilty when when I realize that I consi...        NaN
    7      joy  After my girlfriend had taken her exam we went...        NaN
    8     fear  When, for the first time I realized the meanin...        NaN
    9    anger  When a car is overtaking another and I am forc...        NaN



```python
import seaborn as sns
import tkinter
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
%matplotlib inline
sns.countplot(x="label", data=df)
```




    <AxesSubplot:xlabel='label', ylabel='count'>




    
![png](output_9_1.png)
    



```python
print(sentences[:5])
```

    ['On days when I feel close to my partner and other friends.   \nWhen I feel at peace with myself and also experience a close  \ncontact with people whom I regard greatly.', 'Every time I imagine that someone I love or I could contact a  \nserious illness, even death.', 'When I had been obviously unjustly treated and had no possibility  \nof elucidating this.', 'When I think about the short time that we live and relate it to  \nthe periods of my life when I think that I did not use this  \nshort time.', 'At a gathering I found myself involuntarily sitting next to two  \npeople who expressed opinions that I considered very low and  \ndiscriminating.']



```python
nltk.download("stopwords")
nltk.download("punkt")
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /home/shiningflash/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package punkt to
    [nltk_data]     /home/shiningflash/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!





    True



## 2. Data Cleaning


```python
#define stemmer
stemmer = LancasterStemmer()
```


```python
def cleaning(sentences):
  words = []
  for s in sentences:
    clean = re.sub(r'[^ a-z A-Z 0-9]', " ", s)
    w = word_tokenize(clean)
    words.append([i.lower() for i in w])
    
  return words 
```


```python
cleaned_words = cleaning(sentences)
print(len(cleaned_words))
print(cleaned_words[:2])  
```

    7516
    [['on', 'days', 'when', 'i', 'feel', 'close', 'to', 'my', 'partner', 'and', 'other', 'friends', 'when', 'i', 'feel', 'at', 'peace', 'with', 'myself', 'and', 'also', 'experience', 'a', 'close', 'contact', 'with', 'people', 'whom', 'i', 'regard', 'greatly'], ['every', 'time', 'i', 'imagine', 'that', 'someone', 'i', 'love', 'or', 'i', 'could', 'contact', 'a', 'serious', 'illness', 'even', 'death']]


## 3. Texts Tokenization


```python
def create_tokenizer(words, filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'):
  token = Tokenizer(filters = filters)
  token.fit_on_texts(words)
  return token
```


```python
def max_length(words):
  return(len(max(words, key = len)))
```


```python
word_tokenizer = create_tokenizer(cleaned_words)
vocab_size = len(word_tokenizer.word_index) + 1
max_length = max_length(cleaned_words)

print("Vocab Size = %d and Maximum length = %d" % (vocab_size, max_length))
```

    Vocab Size = 8989 and Maximum length = 179



```python
def encoding_doc(token, words):
  return(token.texts_to_sequences(words))
```


```python
encoded_doc = encoding_doc(word_tokenizer, cleaned_words)
```


```python
def padding_doc(encoded_doc, max_length):
  return(pad_sequences(encoded_doc, maxlen = max_length, padding = "post"))
```


```python
padded_doc = padding_doc(encoded_doc, max_length)
```


```python
print("Shape of padded docs = ",padded_doc.shape)
```

    Shape of padded docs =  (7516, 179)



```python
#tokenizer with filter changed
output_tokenizer = create_tokenizer(unique_label, filters = '!"#$%&()*+,-/:;<=>?@[\]^`{|}~')
```


```python
output_tokenizer.word_index
```




    {'disgust': 1,
     'fear': 2,
     'anger': 3,
     'joy': 4,
     'guilt': 5,
     'shame': 6,
     'sadness': 7}




```python
encoded_output = encoding_doc(output_tokenizer, label)
```


```python
encoded_output = np.array(encoded_output).reshape(len(encoded_output), 1)
```


```python
encoded_output.shape
```




    (7516, 1)




```python
def one_hot(encode):
  o = OneHotEncoder(sparse = False)
  return(o.fit_transform(encode))
```


```python
output_one_hot = one_hot(encoded_output)
```


```python
output_one_hot.shape
```




    (7516, 7)




```python
from sklearn.model_selection import train_test_split
```


```python
train_X, val_X, train_Y, val_Y = train_test_split(padded_doc, output_one_hot, shuffle = True, test_size = 0.2)
```


```python
print("Shape of train_X = %s and train_Y = %s" % (train_X.shape, train_Y.shape))
print("Shape of val_X = %s and val_Y = %s" % (val_X.shape, val_Y.shape))
```

    Shape of train_X = (6012, 179) and train_Y = (6012, 7)
    Shape of val_X = (1504, 179) and val_Y = (1504, 7)


## 4. Bidirectional GRU 


```python
def create_model(vocab_size, max_length):
  model = Sequential()
  model.add(Embedding(vocab_size, 128, input_length = max_length, trainable = False))
  model.add(Bidirectional(GRU(128)))
  model.add(Dense(32, activation = "relu"))
  model.add(Dropout(0.5))
  model.add(Dense(7, activation = "softmax"))
  
  return model
```


```python
model = create_model(vocab_size, max_length)

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (None, 179, 128)          1150592   
    _________________________________________________________________
    bidirectional (Bidirectional (None, 256)               198144    
    _________________________________________________________________
    dense (Dense)                (None, 32)                8224      
    _________________________________________________________________
    dropout (Dropout)            (None, 32)                0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 7)                 231       
    =================================================================
    Total params: 1,357,191
    Trainable params: 206,599
    Non-trainable params: 1,150,592
    _________________________________________________________________



```python
filename = 'model.h5'
checkpoint = ModelCheckpoint(filename,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')
```


```python
hist = model.fit(train_X, train_Y,
                 epochs = 100,
                 batch_size = 32,
                 validation_data = (val_X, val_Y),
                 callbacks = [checkpoint])
```

    Epoch 1/100
    188/188 [==============================] - ETA: 0s - loss: 1.9452 - accuracy: 0.1565
    Epoch 00001: val_loss improved from inf to 1.93838, saving model to model.h5
    188/188 [==============================] - 28s 147ms/step - loss: 1.9452 - accuracy: 0.1565 - val_loss: 1.9384 - val_accuracy: 0.1868
    Epoch 2/100
    188/188 [==============================] - ETA: 0s - loss: 1.9222 - accuracy: 0.1919
    Epoch 00002: val_loss improved from 1.93838 to 1.87944, saving model to model.h5
    188/188 [==============================] - 30s 157ms/step - loss: 1.9222 - accuracy: 0.1919 - val_loss: 1.8794 - val_accuracy: 0.2374
    Epoch 3/100
    188/188 [==============================] - ETA: 0s - loss: 1.8723 - accuracy: 0.2399
    Epoch 00003: val_loss improved from 1.87944 to 1.84961, saving model to model.h5
    188/188 [==============================] - 31s 162ms/step - loss: 1.8723 - accuracy: 0.2399 - val_loss: 1.8496 - val_accuracy: 0.2646
    Epoch 4/100
    188/188 [==============================] - ETA: 0s - loss: 1.8285 - accuracy: 0.2725
    Epoch 00004: val_loss improved from 1.84961 to 1.81347, saving model to model.h5
    188/188 [==============================] - 30s 159ms/step - loss: 1.8285 - accuracy: 0.2725 - val_loss: 1.8135 - val_accuracy: 0.2713
    Epoch 5/100
    188/188 [==============================] - ETA: 0s - loss: 1.7885 - accuracy: 0.3071
    Epoch 00005: val_loss improved from 1.81347 to 1.76444, saving model to model.h5
    188/188 [==============================] - 30s 159ms/step - loss: 1.7885 - accuracy: 0.3071 - val_loss: 1.7644 - val_accuracy: 0.3112
    Epoch 6/100
    188/188 [==============================] - ETA: 0s - loss: 1.7603 - accuracy: 0.3267
    Epoch 00006: val_loss improved from 1.76444 to 1.74935, saving model to model.h5
    188/188 [==============================] - 27s 146ms/step - loss: 1.7603 - accuracy: 0.3267 - val_loss: 1.7494 - val_accuracy: 0.3271
    Epoch 7/100
    188/188 [==============================] - ETA: 0s - loss: 1.7246 - accuracy: 0.3382
    Epoch 00007: val_loss improved from 1.74935 to 1.73108, saving model to model.h5
    188/188 [==============================] - 27s 145ms/step - loss: 1.7246 - accuracy: 0.3382 - val_loss: 1.7311 - val_accuracy: 0.3298
    Epoch 8/100
    188/188 [==============================] - ETA: 0s - loss: 1.7009 - accuracy: 0.3535
    Epoch 00008: val_loss improved from 1.73108 to 1.71289, saving model to model.h5
    188/188 [==============================] - 27s 146ms/step - loss: 1.7009 - accuracy: 0.3535 - val_loss: 1.7129 - val_accuracy: 0.3311
    Epoch 9/100
    188/188 [==============================] - ETA: 0s - loss: 1.6743 - accuracy: 0.3711
    Epoch 00009: val_loss did not improve from 1.71289
    188/188 [==============================] - 28s 152ms/step - loss: 1.6743 - accuracy: 0.3711 - val_loss: 1.7371 - val_accuracy: 0.3191
    Epoch 10/100
    188/188 [==============================] - ETA: 0s - loss: 1.6462 - accuracy: 0.3794
    Epoch 00010: val_loss improved from 1.71289 to 1.69525, saving model to model.h5
    188/188 [==============================] - 32s 170ms/step - loss: 1.6462 - accuracy: 0.3794 - val_loss: 1.6953 - val_accuracy: 0.3484
    Epoch 11/100
    188/188 [==============================] - ETA: 0s - loss: 1.6103 - accuracy: 0.4017
    Epoch 00011: val_loss improved from 1.69525 to 1.66417, saving model to model.h5
    188/188 [==============================] - 31s 166ms/step - loss: 1.6103 - accuracy: 0.4017 - val_loss: 1.6642 - val_accuracy: 0.3517
    Epoch 12/100
    188/188 [==============================] - ETA: 0s - loss: 1.5782 - accuracy: 0.4138
    Epoch 00012: val_loss did not improve from 1.66417
    188/188 [==============================] - 31s 164ms/step - loss: 1.5782 - accuracy: 0.4138 - val_loss: 1.6749 - val_accuracy: 0.3644
    Epoch 13/100
    188/188 [==============================] - ETA: 0s - loss: 1.5537 - accuracy: 0.4213
    Epoch 00013: val_loss improved from 1.66417 to 1.65436, saving model to model.h5
    188/188 [==============================] - 31s 165ms/step - loss: 1.5537 - accuracy: 0.4213 - val_loss: 1.6544 - val_accuracy: 0.3670
    Epoch 14/100
    188/188 [==============================] - ETA: 0s - loss: 1.5358 - accuracy: 0.4366
    Epoch 00014: val_loss improved from 1.65436 to 1.64753, saving model to model.h5
    188/188 [==============================] - 31s 167ms/step - loss: 1.5358 - accuracy: 0.4366 - val_loss: 1.6475 - val_accuracy: 0.3690
    Epoch 15/100
    188/188 [==============================] - ETA: 0s - loss: 1.5015 - accuracy: 0.4481
    Epoch 00015: val_loss improved from 1.64753 to 1.64574, saving model to model.h5
    188/188 [==============================] - 31s 166ms/step - loss: 1.5015 - accuracy: 0.4481 - val_loss: 1.6457 - val_accuracy: 0.3850
    Epoch 16/100
    188/188 [==============================] - ETA: 0s - loss: 1.4612 - accuracy: 0.4656
    Epoch 00016: val_loss improved from 1.64574 to 1.58966, saving model to model.h5
    188/188 [==============================] - 34s 182ms/step - loss: 1.4612 - accuracy: 0.4656 - val_loss: 1.5897 - val_accuracy: 0.3883
    Epoch 17/100
    188/188 [==============================] - ETA: 0s - loss: 1.4441 - accuracy: 0.4632
    Epoch 00017: val_loss did not improve from 1.58966
    188/188 [==============================] - 39s 207ms/step - loss: 1.4441 - accuracy: 0.4632 - val_loss: 1.6062 - val_accuracy: 0.3989
    Epoch 18/100
    188/188 [==============================] - ETA: 0s - loss: 1.4211 - accuracy: 0.4726
    Epoch 00018: val_loss did not improve from 1.58966
    188/188 [==============================] - 33s 174ms/step - loss: 1.4211 - accuracy: 0.4726 - val_loss: 1.6070 - val_accuracy: 0.3983
    Epoch 19/100
    188/188 [==============================] - ETA: 0s - loss: 1.3896 - accuracy: 0.4943
    Epoch 00019: val_loss improved from 1.58966 to 1.58049, saving model to model.h5
    188/188 [==============================] - 51s 269ms/step - loss: 1.3896 - accuracy: 0.4943 - val_loss: 1.5805 - val_accuracy: 0.4062
    Epoch 20/100
    188/188 [==============================] - ETA: 0s - loss: 1.3634 - accuracy: 0.4983
    Epoch 00020: val_loss improved from 1.58049 to 1.57140, saving model to model.h5
    188/188 [==============================] - 34s 183ms/step - loss: 1.3634 - accuracy: 0.4983 - val_loss: 1.5714 - val_accuracy: 0.4069
    Epoch 21/100
    188/188 [==============================] - ETA: 0s - loss: 1.3487 - accuracy: 0.5115
    Epoch 00021: val_loss did not improve from 1.57140
    188/188 [==============================] - 34s 183ms/step - loss: 1.3487 - accuracy: 0.5115 - val_loss: 1.5864 - val_accuracy: 0.4142
    Epoch 22/100
    188/188 [==============================] - ETA: 0s - loss: 1.3191 - accuracy: 0.5156
    Epoch 00022: val_loss improved from 1.57140 to 1.55276, saving model to model.h5
    188/188 [==============================] - 34s 179ms/step - loss: 1.3191 - accuracy: 0.5156 - val_loss: 1.5528 - val_accuracy: 0.4182
    Epoch 23/100
    188/188 [==============================] - ETA: 0s - loss: 1.2795 - accuracy: 0.5351
    Epoch 00023: val_loss did not improve from 1.55276
    188/188 [==============================] - 34s 183ms/step - loss: 1.2795 - accuracy: 0.5351 - val_loss: 1.5809 - val_accuracy: 0.4249
    Epoch 24/100
    188/188 [==============================] - ETA: 0s - loss: 1.2532 - accuracy: 0.5482
    Epoch 00024: val_loss did not improve from 1.55276
    188/188 [==============================] - 34s 179ms/step - loss: 1.2532 - accuracy: 0.5482 - val_loss: 1.5714 - val_accuracy: 0.4335
    Epoch 25/100
    188/188 [==============================] - ETA: 0s - loss: 1.2378 - accuracy: 0.5497
    Epoch 00025: val_loss did not improve from 1.55276
    188/188 [==============================] - 36s 189ms/step - loss: 1.2378 - accuracy: 0.5497 - val_loss: 1.6285 - val_accuracy: 0.4229
    Epoch 26/100
    188/188 [==============================] - ETA: 0s - loss: 1.2097 - accuracy: 0.5612
    Epoch 00026: val_loss did not improve from 1.55276
    188/188 [==============================] - 35s 185ms/step - loss: 1.2097 - accuracy: 0.5612 - val_loss: 1.6163 - val_accuracy: 0.4362
    Epoch 27/100
    188/188 [==============================] - ETA: 0s - loss: 1.1966 - accuracy: 0.5627
    Epoch 00027: val_loss did not improve from 1.55276
    188/188 [==============================] - 32s 172ms/step - loss: 1.1966 - accuracy: 0.5627 - val_loss: 1.5994 - val_accuracy: 0.4322
    Epoch 28/100
    188/188 [==============================] - ETA: 0s - loss: 1.1610 - accuracy: 0.5813
    Epoch 00028: val_loss did not improve from 1.55276
    188/188 [==============================] - 35s 185ms/step - loss: 1.1610 - accuracy: 0.5813 - val_loss: 1.6647 - val_accuracy: 0.4355
    Epoch 29/100
    188/188 [==============================] - ETA: 0s - loss: 1.1672 - accuracy: 0.5670
    Epoch 00029: val_loss did not improve from 1.55276
    188/188 [==============================] - 33s 175ms/step - loss: 1.1672 - accuracy: 0.5670 - val_loss: 1.6053 - val_accuracy: 0.4355
    Epoch 30/100
    188/188 [==============================] - ETA: 0s - loss: 1.1150 - accuracy: 0.5973
    Epoch 00030: val_loss did not improve from 1.55276
    188/188 [==============================] - 32s 172ms/step - loss: 1.1150 - accuracy: 0.5973 - val_loss: 1.6671 - val_accuracy: 0.4282
    Epoch 31/100
    188/188 [==============================] - ETA: 0s - loss: 1.0833 - accuracy: 0.6006
    Epoch 00031: val_loss did not improve from 1.55276
    188/188 [==============================] - 32s 170ms/step - loss: 1.0833 - accuracy: 0.6006 - val_loss: 1.7245 - val_accuracy: 0.4202
    Epoch 32/100
    188/188 [==============================] - ETA: 0s - loss: 1.0542 - accuracy: 0.6199
    Epoch 00032: val_loss did not improve from 1.55276
    188/188 [==============================] - 33s 174ms/step - loss: 1.0542 - accuracy: 0.6199 - val_loss: 1.7344 - val_accuracy: 0.4422
    Epoch 33/100
    188/188 [==============================] - ETA: 0s - loss: 1.0196 - accuracy: 0.6219
    Epoch 00033: val_loss did not improve from 1.55276
    188/188 [==============================] - 33s 177ms/step - loss: 1.0196 - accuracy: 0.6219 - val_loss: 1.7784 - val_accuracy: 0.4335
    Epoch 34/100
    188/188 [==============================] - ETA: 0s - loss: 0.9958 - accuracy: 0.6347
    Epoch 00034: val_loss did not improve from 1.55276
    188/188 [==============================] - 34s 183ms/step - loss: 0.9958 - accuracy: 0.6347 - val_loss: 1.7854 - val_accuracy: 0.4448
    Epoch 35/100
    188/188 [==============================] - ETA: 0s - loss: 1.0154 - accuracy: 0.6357
    Epoch 00035: val_loss did not improve from 1.55276
    188/188 [==============================] - 33s 175ms/step - loss: 1.0154 - accuracy: 0.6357 - val_loss: 1.6758 - val_accuracy: 0.4375
    Epoch 36/100
    188/188 [==============================] - ETA: 0s - loss: 0.9666 - accuracy: 0.6439
    Epoch 00036: val_loss did not improve from 1.55276
    188/188 [==============================] - 34s 178ms/step - loss: 0.9666 - accuracy: 0.6439 - val_loss: 1.8552 - val_accuracy: 0.4229
    Epoch 37/100
    188/188 [==============================] - ETA: 0s - loss: 0.9357 - accuracy: 0.6530
    Epoch 00037: val_loss did not improve from 1.55276
    188/188 [==============================] - 35s 188ms/step - loss: 0.9357 - accuracy: 0.6530 - val_loss: 1.8407 - val_accuracy: 0.4295
    Epoch 38/100
    188/188 [==============================] - ETA: 0s - loss: 0.9160 - accuracy: 0.6612
    Epoch 00038: val_loss did not improve from 1.55276
    188/188 [==============================] - 34s 179ms/step - loss: 0.9160 - accuracy: 0.6612 - val_loss: 1.8459 - val_accuracy: 0.4269
    Epoch 39/100
    188/188 [==============================] - ETA: 0s - loss: 0.8882 - accuracy: 0.6732
    Epoch 00039: val_loss did not improve from 1.55276
    188/188 [==============================] - 34s 179ms/step - loss: 0.8882 - accuracy: 0.6732 - val_loss: 2.0239 - val_accuracy: 0.4322
    Epoch 40/100
    188/188 [==============================] - ETA: 0s - loss: 0.8747 - accuracy: 0.6761
    Epoch 00040: val_loss did not improve from 1.55276
    188/188 [==============================] - 32s 171ms/step - loss: 0.8747 - accuracy: 0.6761 - val_loss: 2.1101 - val_accuracy: 0.4269
    Epoch 41/100
    188/188 [==============================] - ETA: 0s - loss: 0.8317 - accuracy: 0.6979
    Epoch 00041: val_loss did not improve from 1.55276
    188/188 [==============================] - 28s 149ms/step - loss: 0.8317 - accuracy: 0.6979 - val_loss: 2.0988 - val_accuracy: 0.4295
    Epoch 42/100
    188/188 [==============================] - ETA: 0s - loss: 0.8136 - accuracy: 0.6938
    Epoch 00042: val_loss did not improve from 1.55276
    188/188 [==============================] - 27s 146ms/step - loss: 0.8136 - accuracy: 0.6938 - val_loss: 2.1724 - val_accuracy: 0.4269
    Epoch 43/100
    188/188 [==============================] - ETA: 0s - loss: 0.8213 - accuracy: 0.6913
    Epoch 00043: val_loss did not improve from 1.55276
    188/188 [==============================] - 27s 146ms/step - loss: 0.8213 - accuracy: 0.6913 - val_loss: 2.1551 - val_accuracy: 0.4302
    Epoch 44/100
    188/188 [==============================] - ETA: 0s - loss: 0.8042 - accuracy: 0.7021
    Epoch 00044: val_loss did not improve from 1.55276
    188/188 [==============================] - 28s 148ms/step - loss: 0.8042 - accuracy: 0.7021 - val_loss: 2.2888 - val_accuracy: 0.4229
    Epoch 45/100
    188/188 [==============================] - ETA: 0s - loss: 0.7735 - accuracy: 0.7109
    Epoch 00045: val_loss did not improve from 1.55276
    188/188 [==============================] - 28s 148ms/step - loss: 0.7735 - accuracy: 0.7109 - val_loss: 2.2802 - val_accuracy: 0.4176
    Epoch 46/100
    188/188 [==============================] - ETA: 0s - loss: 0.7639 - accuracy: 0.7139
    Epoch 00046: val_loss did not improve from 1.55276
    188/188 [==============================] - 28s 147ms/step - loss: 0.7639 - accuracy: 0.7139 - val_loss: 2.2815 - val_accuracy: 0.4096
    Epoch 47/100
    188/188 [==============================] - ETA: 0s - loss: 0.7462 - accuracy: 0.7186
    Epoch 00047: val_loss did not improve from 1.55276
    188/188 [==============================] - 28s 148ms/step - loss: 0.7462 - accuracy: 0.7186 - val_loss: 2.4362 - val_accuracy: 0.4202
    Epoch 48/100
    188/188 [==============================] - ETA: 0s - loss: 0.7047 - accuracy: 0.7305
    Epoch 00048: val_loss did not improve from 1.55276
    188/188 [==============================] - 29s 156ms/step - loss: 0.7047 - accuracy: 0.7305 - val_loss: 2.5796 - val_accuracy: 0.4142
    Epoch 49/100
    188/188 [==============================] - ETA: 0s - loss: 0.6828 - accuracy: 0.7387
    Epoch 00049: val_loss did not improve from 1.55276
    188/188 [==============================] - 28s 150ms/step - loss: 0.6828 - accuracy: 0.7387 - val_loss: 2.6442 - val_accuracy: 0.4215
    Epoch 50/100
    188/188 [==============================] - ETA: 0s - loss: 0.7265 - accuracy: 0.7277
    Epoch 00050: val_loss did not improve from 1.55276
    188/188 [==============================] - 28s 148ms/step - loss: 0.7265 - accuracy: 0.7277 - val_loss: 2.3500 - val_accuracy: 0.4156
    Epoch 51/100
    188/188 [==============================] - ETA: 0s - loss: 0.7387 - accuracy: 0.7214
    Epoch 00051: val_loss did not improve from 1.55276
    188/188 [==============================] - 28s 149ms/step - loss: 0.7387 - accuracy: 0.7214 - val_loss: 2.4236 - val_accuracy: 0.4249
    Epoch 52/100
    188/188 [==============================] - ETA: 0s - loss: 0.6672 - accuracy: 0.7458
    Epoch 00052: val_loss did not improve from 1.55276
    188/188 [==============================] - 28s 151ms/step - loss: 0.6672 - accuracy: 0.7458 - val_loss: 2.7802 - val_accuracy: 0.4195
    Epoch 53/100
    188/188 [==============================] - ETA: 0s - loss: 0.6545 - accuracy: 0.7525
    Epoch 00053: val_loss did not improve from 1.55276
    188/188 [==============================] - 29s 153ms/step - loss: 0.6545 - accuracy: 0.7525 - val_loss: 2.9663 - val_accuracy: 0.4162
    Epoch 54/100
    188/188 [==============================] - ETA: 0s - loss: 0.6288 - accuracy: 0.7540
    Epoch 00054: val_loss did not improve from 1.55276
    188/188 [==============================] - 28s 150ms/step - loss: 0.6288 - accuracy: 0.7540 - val_loss: 2.9011 - val_accuracy: 0.4122
    Epoch 55/100
    188/188 [==============================] - ETA: 0s - loss: 0.6045 - accuracy: 0.7736
    Epoch 00055: val_loss did not improve from 1.55276
    188/188 [==============================] - 28s 149ms/step - loss: 0.6045 - accuracy: 0.7736 - val_loss: 2.7860 - val_accuracy: 0.4289
    Epoch 56/100
    188/188 [==============================] - ETA: 0s - loss: 0.5864 - accuracy: 0.7720
    Epoch 00056: val_loss did not improve from 1.55276
    188/188 [==============================] - 28s 148ms/step - loss: 0.5864 - accuracy: 0.7720 - val_loss: 2.8519 - val_accuracy: 0.4076
    Epoch 57/100
    188/188 [==============================] - ETA: 0s - loss: 0.6227 - accuracy: 0.7636
    Epoch 00057: val_loss did not improve from 1.55276
    188/188 [==============================] - 28s 147ms/step - loss: 0.6227 - accuracy: 0.7636 - val_loss: 2.9346 - val_accuracy: 0.4195
    Epoch 58/100
    188/188 [==============================] - ETA: 0s - loss: 0.5814 - accuracy: 0.7754
    Epoch 00058: val_loss did not improve from 1.55276
    188/188 [==============================] - 28s 151ms/step - loss: 0.5814 - accuracy: 0.7754 - val_loss: 3.0307 - val_accuracy: 0.4182
    Epoch 59/100
    188/188 [==============================] - ETA: 0s - loss: 0.5541 - accuracy: 0.7884
    Epoch 00059: val_loss did not improve from 1.55276
    188/188 [==============================] - 28s 149ms/step - loss: 0.5541 - accuracy: 0.7884 - val_loss: 3.0622 - val_accuracy: 0.4189
    Epoch 60/100
    188/188 [==============================] - ETA: 0s - loss: 0.5720 - accuracy: 0.7824
    Epoch 00060: val_loss did not improve from 1.55276
    188/188 [==============================] - 28s 147ms/step - loss: 0.5720 - accuracy: 0.7824 - val_loss: 3.1490 - val_accuracy: 0.4189
    Epoch 61/100
    188/188 [==============================] - ETA: 0s - loss: 0.5209 - accuracy: 0.7939
    Epoch 00061: val_loss did not improve from 1.55276
    188/188 [==============================] - 28s 147ms/step - loss: 0.5209 - accuracy: 0.7939 - val_loss: 2.9995 - val_accuracy: 0.4162
    Epoch 62/100
    188/188 [==============================] - ETA: 0s - loss: 0.5315 - accuracy: 0.7984
    Epoch 00062: val_loss did not improve from 1.55276
    188/188 [==============================] - 28s 148ms/step - loss: 0.5315 - accuracy: 0.7984 - val_loss: 2.9712 - val_accuracy: 0.4262
    Epoch 63/100
    188/188 [==============================] - ETA: 0s - loss: 0.5173 - accuracy: 0.8037
    Epoch 00063: val_loss did not improve from 1.55276
    188/188 [==============================] - 27s 146ms/step - loss: 0.5173 - accuracy: 0.8037 - val_loss: 3.1873 - val_accuracy: 0.4089
    Epoch 64/100
    188/188 [==============================] - ETA: 0s - loss: 0.4850 - accuracy: 0.8155
    Epoch 00064: val_loss did not improve from 1.55276
    188/188 [==============================] - 27s 146ms/step - loss: 0.4850 - accuracy: 0.8155 - val_loss: 3.6163 - val_accuracy: 0.4142
    Epoch 65/100
    188/188 [==============================] - ETA: 0s - loss: 0.4965 - accuracy: 0.8107
    Epoch 00065: val_loss did not improve from 1.55276
    188/188 [==============================] - 34s 183ms/step - loss: 0.4965 - accuracy: 0.8107 - val_loss: 3.4945 - val_accuracy: 0.4156
    Epoch 66/100
    188/188 [==============================] - ETA: 0s - loss: 0.4581 - accuracy: 0.8217
    Epoch 00066: val_loss did not improve from 1.55276
    188/188 [==============================] - 34s 180ms/step - loss: 0.4581 - accuracy: 0.8217 - val_loss: 3.4051 - val_accuracy: 0.4176
    Epoch 67/100
    188/188 [==============================] - ETA: 0s - loss: 0.4579 - accuracy: 0.8273
    Epoch 00067: val_loss did not improve from 1.55276
    188/188 [==============================] - 39s 207ms/step - loss: 0.4579 - accuracy: 0.8273 - val_loss: 3.4118 - val_accuracy: 0.4149
    Epoch 68/100
    188/188 [==============================] - ETA: 0s - loss: 0.5098 - accuracy: 0.8117
    Epoch 00068: val_loss did not improve from 1.55276
    188/188 [==============================] - 37s 194ms/step - loss: 0.5098 - accuracy: 0.8117 - val_loss: 3.2042 - val_accuracy: 0.4142
    Epoch 69/100
    188/188 [==============================] - ETA: 0s - loss: 0.4947 - accuracy: 0.8150
    Epoch 00069: val_loss did not improve from 1.55276
    188/188 [==============================] - 33s 173ms/step - loss: 0.4947 - accuracy: 0.8150 - val_loss: 3.3984 - val_accuracy: 0.4169
    Epoch 70/100
    188/188 [==============================] - ETA: 0s - loss: 0.4274 - accuracy: 0.8317
    Epoch 00070: val_loss did not improve from 1.55276
    188/188 [==============================] - 31s 164ms/step - loss: 0.4274 - accuracy: 0.8317 - val_loss: 3.8017 - val_accuracy: 0.4222
    Epoch 71/100
    188/188 [==============================] - ETA: 0s - loss: 0.4045 - accuracy: 0.8441
    Epoch 00071: val_loss did not improve from 1.55276
    188/188 [==============================] - 31s 163ms/step - loss: 0.4045 - accuracy: 0.8441 - val_loss: 3.8804 - val_accuracy: 0.4043
    Epoch 72/100
    188/188 [==============================] - ETA: 0s - loss: 0.4158 - accuracy: 0.8405
    Epoch 00072: val_loss did not improve from 1.55276
    188/188 [==============================] - 32s 169ms/step - loss: 0.4158 - accuracy: 0.8405 - val_loss: 3.8641 - val_accuracy: 0.4069
    Epoch 73/100
    188/188 [==============================] - ETA: 0s - loss: 0.4346 - accuracy: 0.8345
    Epoch 00073: val_loss did not improve from 1.55276
    188/188 [==============================] - 32s 168ms/step - loss: 0.4346 - accuracy: 0.8345 - val_loss: 3.7514 - val_accuracy: 0.4076
    Epoch 74/100
    188/188 [==============================] - ETA: 0s - loss: 0.4065 - accuracy: 0.8405
    Epoch 00074: val_loss did not improve from 1.55276
    188/188 [==============================] - 28s 150ms/step - loss: 0.4065 - accuracy: 0.8405 - val_loss: 3.6998 - val_accuracy: 0.4149
    Epoch 75/100
    188/188 [==============================] - ETA: 0s - loss: 0.4105 - accuracy: 0.8455
    Epoch 00075: val_loss did not improve from 1.55276
    188/188 [==============================] - 28s 147ms/step - loss: 0.4105 - accuracy: 0.8455 - val_loss: 4.2083 - val_accuracy: 0.4182
    Epoch 76/100
    188/188 [==============================] - ETA: 0s - loss: 0.7065 - accuracy: 0.7507
    Epoch 00076: val_loss did not improve from 1.55276
    188/188 [==============================] - 28s 147ms/step - loss: 0.7065 - accuracy: 0.7507 - val_loss: 3.2775 - val_accuracy: 0.4162
    Epoch 77/100
    188/188 [==============================] - ETA: 0s - loss: 0.5469 - accuracy: 0.8009
    Epoch 00077: val_loss did not improve from 1.55276
    188/188 [==============================] - 30s 157ms/step - loss: 0.5469 - accuracy: 0.8009 - val_loss: 3.5154 - val_accuracy: 0.4156
    Epoch 78/100
    188/188 [==============================] - ETA: 0s - loss: 0.4297 - accuracy: 0.8385
    Epoch 00078: val_loss did not improve from 1.55276
    188/188 [==============================] - 32s 170ms/step - loss: 0.4297 - accuracy: 0.8385 - val_loss: 3.8071 - val_accuracy: 0.4182
    Epoch 79/100
    188/188 [==============================] - ETA: 0s - loss: 0.3922 - accuracy: 0.8506
    Epoch 00079: val_loss did not improve from 1.55276
    188/188 [==============================] - 32s 169ms/step - loss: 0.3922 - accuracy: 0.8506 - val_loss: 4.0850 - val_accuracy: 0.4169
    Epoch 80/100
    188/188 [==============================] - ETA: 0s - loss: 0.3451 - accuracy: 0.8669
    Epoch 00080: val_loss did not improve from 1.55276
    188/188 [==============================] - 34s 182ms/step - loss: 0.3451 - accuracy: 0.8669 - val_loss: 4.1809 - val_accuracy: 0.4176
    Epoch 81/100
    188/188 [==============================] - ETA: 0s - loss: 0.3807 - accuracy: 0.8588
    Epoch 00081: val_loss did not improve from 1.55276
    188/188 [==============================] - 29s 156ms/step - loss: 0.3807 - accuracy: 0.8588 - val_loss: 3.8282 - val_accuracy: 0.4182
    Epoch 82/100
    188/188 [==============================] - ETA: 0s - loss: 0.3530 - accuracy: 0.8673
    Epoch 00082: val_loss did not improve from 1.55276
    188/188 [==============================] - 28s 150ms/step - loss: 0.3530 - accuracy: 0.8673 - val_loss: 4.4239 - val_accuracy: 0.4249
    Epoch 83/100
    188/188 [==============================] - ETA: 0s - loss: 0.3249 - accuracy: 0.8796
    Epoch 00083: val_loss did not improve from 1.55276
    188/188 [==============================] - 28s 146ms/step - loss: 0.3249 - accuracy: 0.8796 - val_loss: 4.5924 - val_accuracy: 0.4102
    Epoch 84/100
    188/188 [==============================] - ETA: 0s - loss: 0.3642 - accuracy: 0.8629
    Epoch 00084: val_loss did not improve from 1.55276
    188/188 [==============================] - 30s 159ms/step - loss: 0.3642 - accuracy: 0.8629 - val_loss: 4.5836 - val_accuracy: 0.4129
    Epoch 85/100
    188/188 [==============================] - ETA: 0s - loss: 0.3277 - accuracy: 0.8731
    Epoch 00085: val_loss did not improve from 1.55276
    188/188 [==============================] - 31s 168ms/step - loss: 0.3277 - accuracy: 0.8731 - val_loss: 4.6225 - val_accuracy: 0.4156
    Epoch 86/100
    188/188 [==============================] - ETA: 0s - loss: 0.3257 - accuracy: 0.8734
    Epoch 00086: val_loss did not improve from 1.55276
    188/188 [==============================] - 30s 159ms/step - loss: 0.3257 - accuracy: 0.8734 - val_loss: 4.2445 - val_accuracy: 0.4315
    Epoch 87/100
    188/188 [==============================] - ETA: 0s - loss: 0.3569 - accuracy: 0.8648
    Epoch 00087: val_loss did not improve from 1.55276
    188/188 [==============================] - 30s 160ms/step - loss: 0.3569 - accuracy: 0.8648 - val_loss: 4.5867 - val_accuracy: 0.4129
    Epoch 88/100
    188/188 [==============================] - ETA: 0s - loss: 0.2930 - accuracy: 0.8861
    Epoch 00088: val_loss did not improve from 1.55276
    188/188 [==============================] - 31s 162ms/step - loss: 0.2930 - accuracy: 0.8861 - val_loss: 4.7156 - val_accuracy: 0.4235
    Epoch 89/100
    188/188 [==============================] - ETA: 0s - loss: 0.2728 - accuracy: 0.8970
    Epoch 00089: val_loss did not improve from 1.55276
    188/188 [==============================] - 31s 163ms/step - loss: 0.2728 - accuracy: 0.8970 - val_loss: 4.6431 - val_accuracy: 0.4129
    Epoch 90/100
    188/188 [==============================] - ETA: 0s - loss: 0.2950 - accuracy: 0.8892
    Epoch 00090: val_loss did not improve from 1.55276
    188/188 [==============================] - 30s 161ms/step - loss: 0.2950 - accuracy: 0.8892 - val_loss: 4.9129 - val_accuracy: 0.4176
    Epoch 91/100
    188/188 [==============================] - ETA: 0s - loss: 0.2938 - accuracy: 0.8884
    Epoch 00091: val_loss did not improve from 1.55276
    188/188 [==============================] - 30s 162ms/step - loss: 0.2938 - accuracy: 0.8884 - val_loss: 4.5629 - val_accuracy: 0.4202
    Epoch 92/100
    188/188 [==============================] - ETA: 0s - loss: 0.2748 - accuracy: 0.8930
    Epoch 00092: val_loss did not improve from 1.55276
    188/188 [==============================] - 33s 174ms/step - loss: 0.2748 - accuracy: 0.8930 - val_loss: 5.2193 - val_accuracy: 0.4169
    Epoch 93/100
    188/188 [==============================] - ETA: 0s - loss: 0.3525 - accuracy: 0.8718
    Epoch 00093: val_loss did not improve from 1.55276
    188/188 [==============================] - 31s 167ms/step - loss: 0.3525 - accuracy: 0.8718 - val_loss: 4.3716 - val_accuracy: 0.4162
    Epoch 94/100
    188/188 [==============================] - ETA: 0s - loss: 0.3279 - accuracy: 0.8784
    Epoch 00094: val_loss did not improve from 1.55276
    188/188 [==============================] - 31s 163ms/step - loss: 0.3279 - accuracy: 0.8784 - val_loss: 5.3316 - val_accuracy: 0.4189
    Epoch 95/100
    188/188 [==============================] - ETA: 0s - loss: 0.2808 - accuracy: 0.8940
    Epoch 00095: val_loss did not improve from 1.55276
    188/188 [==============================] - 31s 164ms/step - loss: 0.2808 - accuracy: 0.8940 - val_loss: 5.1539 - val_accuracy: 0.4149
    Epoch 96/100
    188/188 [==============================] - ETA: 0s - loss: 0.2624 - accuracy: 0.9010
    Epoch 00096: val_loss did not improve from 1.55276
    188/188 [==============================] - 31s 165ms/step - loss: 0.2624 - accuracy: 0.9010 - val_loss: 5.4128 - val_accuracy: 0.4082
    Epoch 97/100
    188/188 [==============================] - ETA: 0s - loss: 0.2964 - accuracy: 0.8947
    Epoch 00097: val_loss did not improve from 1.55276
    188/188 [==============================] - 31s 167ms/step - loss: 0.2964 - accuracy: 0.8947 - val_loss: 5.0522 - val_accuracy: 0.4215
    Epoch 98/100
    188/188 [==============================] - ETA: 0s - loss: 0.2692 - accuracy: 0.8987
    Epoch 00098: val_loss did not improve from 1.55276
    188/188 [==============================] - 31s 166ms/step - loss: 0.2692 - accuracy: 0.8987 - val_loss: 5.1637 - val_accuracy: 0.4348
    Epoch 99/100
    188/188 [==============================] - ETA: 0s - loss: 0.2715 - accuracy: 0.8982
    Epoch 00099: val_loss did not improve from 1.55276
    188/188 [==============================] - 31s 165ms/step - loss: 0.2715 - accuracy: 0.8982 - val_loss: 5.0955 - val_accuracy: 0.4242
    Epoch 100/100
    188/188 [==============================] - ETA: 0s - loss: 0.2366 - accuracy: 0.9083
    Epoch 00100: val_loss did not improve from 1.55276
    188/188 [==============================] - 32s 169ms/step - loss: 0.2366 - accuracy: 0.9083 - val_loss: 5.4673 - val_accuracy: 0.4202


## 5. Bidirectional LSTM 


```python
def create_model(vocab_size, max_length):
  model = Sequential()
  model.add(Embedding(vocab_size, 128, input_length = max_length, trainable = False))
  model.add(Bidirectional(LSTM(128)))
  model.add(Dense(32, activation = "relu"))
  model.add(Dropout(0.5))
  model.add(Dense(7, activation = "softmax"))
  
  return model

model_lstm = create_model(vocab_size, max_length)

model_lstm.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model_lstm.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_2 (Embedding)      (None, 179, 128)          1150592   
    _________________________________________________________________
    bidirectional_2 (Bidirection (None, 256)               263168    
    _________________________________________________________________
    dense_4 (Dense)              (None, 32)                8224      
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 32)                0         
    _________________________________________________________________
    dense_5 (Dense)              (None, 7)                 231       
    =================================================================
    Total params: 1,422,215
    Trainable params: 271,623
    Non-trainable params: 1,150,592
    _________________________________________________________________



```python
filename = 'model_lstm.h5'
checkpoint = ModelCheckpoint(filename,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')

hist = model_lstm.fit(train_X, train_Y,
                 epochs = 100,
                 batch_size = 32,
                 validation_data = (val_X, val_Y),
                 callbacks = [checkpoint])
```

    Epoch 1/100
    188/188 [==============================] - ETA: 0s - loss: 1.9454 - accuracy: 0.1504
    Epoch 00001: val_loss improved from inf to 1.93586, saving model to model_lstm.h5
    188/188 [==============================] - 42s 223ms/step - loss: 1.9454 - accuracy: 0.1504 - val_loss: 1.9359 - val_accuracy: 0.1822
    Epoch 2/100
    188/188 [==============================] - ETA: 0s - loss: 1.9173 - accuracy: 0.1964
    Epoch 00002: val_loss improved from 1.93586 to 1.90459, saving model to model_lstm.h5
    188/188 [==============================] - 39s 207ms/step - loss: 1.9173 - accuracy: 0.1964 - val_loss: 1.9046 - val_accuracy: 0.2367
    Epoch 3/100
    188/188 [==============================] - ETA: 0s - loss: 1.8735 - accuracy: 0.2442
    Epoch 00003: val_loss improved from 1.90459 to 1.83846, saving model to model_lstm.h5
    188/188 [==============================] - 38s 203ms/step - loss: 1.8735 - accuracy: 0.2442 - val_loss: 1.8385 - val_accuracy: 0.2819
    Epoch 4/100
    188/188 [==============================] - ETA: 0s - loss: 1.8333 - accuracy: 0.2641
    Epoch 00004: val_loss improved from 1.83846 to 1.83116, saving model to model_lstm.h5
    188/188 [==============================] - 38s 202ms/step - loss: 1.8333 - accuracy: 0.2641 - val_loss: 1.8312 - val_accuracy: 0.2799
    Epoch 5/100
    188/188 [==============================] - ETA: 0s - loss: 1.8033 - accuracy: 0.2946
    Epoch 00005: val_loss improved from 1.83116 to 1.79113, saving model to model_lstm.h5
    188/188 [==============================] - 41s 219ms/step - loss: 1.8033 - accuracy: 0.2946 - val_loss: 1.7911 - val_accuracy: 0.3138
    Epoch 6/100
    188/188 [==============================] - ETA: 0s - loss: 1.7697 - accuracy: 0.3090
    Epoch 00006: val_loss improved from 1.79113 to 1.74590, saving model to model_lstm.h5
    188/188 [==============================] - 39s 206ms/step - loss: 1.7697 - accuracy: 0.3090 - val_loss: 1.7459 - val_accuracy: 0.3424
    Epoch 7/100
    188/188 [==============================] - ETA: 0s - loss: 1.7403 - accuracy: 0.3265
    Epoch 00007: val_loss did not improve from 1.74590
    188/188 [==============================] - 39s 209ms/step - loss: 1.7403 - accuracy: 0.3265 - val_loss: 1.7520 - val_accuracy: 0.3338
    Epoch 8/100
    188/188 [==============================] - ETA: 0s - loss: 1.7114 - accuracy: 0.3405
    Epoch 00008: val_loss improved from 1.74590 to 1.70220, saving model to model_lstm.h5
    188/188 [==============================] - 40s 212ms/step - loss: 1.7114 - accuracy: 0.3405 - val_loss: 1.7022 - val_accuracy: 0.3564
    Epoch 9/100
    188/188 [==============================] - ETA: 0s - loss: 1.7101 - accuracy: 0.3533
    Epoch 00009: val_loss did not improve from 1.70220
    188/188 [==============================] - 40s 212ms/step - loss: 1.7101 - accuracy: 0.3533 - val_loss: 1.7238 - val_accuracy: 0.3324
    Epoch 10/100
    188/188 [==============================] - ETA: 0s - loss: 1.6821 - accuracy: 0.3639
    Epoch 00010: val_loss did not improve from 1.70220
    188/188 [==============================] - 40s 215ms/step - loss: 1.6821 - accuracy: 0.3639 - val_loss: 1.7372 - val_accuracy: 0.3185
    Epoch 11/100
    188/188 [==============================] - ETA: 0s - loss: 1.6686 - accuracy: 0.3661
    Epoch 00011: val_loss did not improve from 1.70220
    188/188 [==============================] - 41s 220ms/step - loss: 1.6686 - accuracy: 0.3661 - val_loss: 1.7133 - val_accuracy: 0.3451
    Epoch 12/100
    188/188 [==============================] - ETA: 0s - loss: 1.6208 - accuracy: 0.3894
    Epoch 00012: val_loss improved from 1.70220 to 1.66652, saving model to model_lstm.h5
    188/188 [==============================] - 41s 217ms/step - loss: 1.6208 - accuracy: 0.3894 - val_loss: 1.6665 - val_accuracy: 0.3517
    Epoch 13/100
    188/188 [==============================] - ETA: 0s - loss: 1.6045 - accuracy: 0.3975
    Epoch 00013: val_loss did not improve from 1.66652
    188/188 [==============================] - 40s 212ms/step - loss: 1.6045 - accuracy: 0.3975 - val_loss: 1.6846 - val_accuracy: 0.3557
    Epoch 14/100
    188/188 [==============================] - ETA: 0s - loss: 1.6139 - accuracy: 0.3962
    Epoch 00014: val_loss did not improve from 1.66652
    188/188 [==============================] - 40s 213ms/step - loss: 1.6139 - accuracy: 0.3962 - val_loss: 1.6942 - val_accuracy: 0.3364
    Epoch 15/100
    188/188 [==============================] - ETA: 0s - loss: 1.6041 - accuracy: 0.3944
    Epoch 00015: val_loss improved from 1.66652 to 1.64109, saving model to model_lstm.h5
    188/188 [==============================] - 41s 215ms/step - loss: 1.6041 - accuracy: 0.3944 - val_loss: 1.6411 - val_accuracy: 0.3757
    Epoch 16/100
    188/188 [==============================] - ETA: 0s - loss: 1.5683 - accuracy: 0.4080
    Epoch 00016: val_loss did not improve from 1.64109
    188/188 [==============================] - 40s 215ms/step - loss: 1.5683 - accuracy: 0.4080 - val_loss: 1.6630 - val_accuracy: 0.3637
    Epoch 17/100
    188/188 [==============================] - ETA: 0s - loss: 1.5475 - accuracy: 0.4283
    Epoch 00017: val_loss improved from 1.64109 to 1.62630, saving model to model_lstm.h5
    188/188 [==============================] - 40s 215ms/step - loss: 1.5475 - accuracy: 0.4283 - val_loss: 1.6263 - val_accuracy: 0.3790
    Epoch 18/100
    188/188 [==============================] - ETA: 0s - loss: 1.5236 - accuracy: 0.4333
    Epoch 00018: val_loss did not improve from 1.62630
    188/188 [==============================] - 39s 208ms/step - loss: 1.5236 - accuracy: 0.4333 - val_loss: 1.6329 - val_accuracy: 0.3783
    Epoch 19/100
    188/188 [==============================] - ETA: 0s - loss: 1.5874 - accuracy: 0.4192
    Epoch 00019: val_loss did not improve from 1.62630
    188/188 [==============================] - 39s 209ms/step - loss: 1.5874 - accuracy: 0.4192 - val_loss: 1.6561 - val_accuracy: 0.3677
    Epoch 20/100
    188/188 [==============================] - ETA: 0s - loss: 1.5186 - accuracy: 0.4315
    Epoch 00020: val_loss did not improve from 1.62630
    188/188 [==============================] - 40s 212ms/step - loss: 1.5186 - accuracy: 0.4315 - val_loss: 1.6457 - val_accuracy: 0.3797
    Epoch 21/100
    188/188 [==============================] - ETA: 0s - loss: 1.4913 - accuracy: 0.4469
    Epoch 00021: val_loss did not improve from 1.62630
    188/188 [==============================] - 40s 212ms/step - loss: 1.4913 - accuracy: 0.4469 - val_loss: 1.6437 - val_accuracy: 0.3830
    Epoch 22/100
    188/188 [==============================] - ETA: 0s - loss: 1.4646 - accuracy: 0.4519
    Epoch 00022: val_loss did not improve from 1.62630
    188/188 [==============================] - 40s 214ms/step - loss: 1.4646 - accuracy: 0.4519 - val_loss: 1.6450 - val_accuracy: 0.3737
    Epoch 23/100
    188/188 [==============================] - ETA: 0s - loss: 1.4751 - accuracy: 0.4456
    Epoch 00023: val_loss did not improve from 1.62630
    188/188 [==============================] - 39s 206ms/step - loss: 1.4751 - accuracy: 0.4456 - val_loss: 1.6344 - val_accuracy: 0.3856
    Epoch 24/100
    188/188 [==============================] - ETA: 0s - loss: 1.4492 - accuracy: 0.4621
    Epoch 00024: val_loss did not improve from 1.62630
    188/188 [==============================] - 39s 206ms/step - loss: 1.4492 - accuracy: 0.4621 - val_loss: 1.6499 - val_accuracy: 0.4003
    Epoch 25/100
    188/188 [==============================] - ETA: 0s - loss: 1.4213 - accuracy: 0.4716
    Epoch 00025: val_loss did not improve from 1.62630
    188/188 [==============================] - 39s 206ms/step - loss: 1.4213 - accuracy: 0.4716 - val_loss: 1.6439 - val_accuracy: 0.3963
    Epoch 26/100
    188/188 [==============================] - ETA: 0s - loss: 1.4046 - accuracy: 0.4779
    Epoch 00026: val_loss improved from 1.62630 to 1.61430, saving model to model_lstm.h5
    188/188 [==============================] - 40s 211ms/step - loss: 1.4046 - accuracy: 0.4779 - val_loss: 1.6143 - val_accuracy: 0.3870
    Epoch 27/100
    188/188 [==============================] - ETA: 0s - loss: 1.4033 - accuracy: 0.4804
    Epoch 00027: val_loss did not improve from 1.61430
    188/188 [==============================] - 41s 217ms/step - loss: 1.4033 - accuracy: 0.4804 - val_loss: 1.6248 - val_accuracy: 0.3956
    Epoch 28/100
    188/188 [==============================] - ETA: 0s - loss: 1.3675 - accuracy: 0.4963
    Epoch 00028: val_loss did not improve from 1.61430
    188/188 [==============================] - 46s 247ms/step - loss: 1.3675 - accuracy: 0.4963 - val_loss: 1.6497 - val_accuracy: 0.3863
    Epoch 29/100
    188/188 [==============================] - ETA: 0s - loss: 1.3540 - accuracy: 0.5035
    Epoch 00029: val_loss did not improve from 1.61430
    188/188 [==============================] - 44s 234ms/step - loss: 1.3540 - accuracy: 0.5035 - val_loss: 1.7095 - val_accuracy: 0.3850
    Epoch 30/100
    188/188 [==============================] - ETA: 0s - loss: 1.3552 - accuracy: 0.4960
    Epoch 00030: val_loss did not improve from 1.61430
    188/188 [==============================] - 43s 228ms/step - loss: 1.3552 - accuracy: 0.4960 - val_loss: 1.6465 - val_accuracy: 0.3949
    Epoch 31/100
    188/188 [==============================] - ETA: 0s - loss: 1.3172 - accuracy: 0.5093
    Epoch 00031: val_loss did not improve from 1.61430
    188/188 [==============================] - 47s 250ms/step - loss: 1.3172 - accuracy: 0.5093 - val_loss: 1.6255 - val_accuracy: 0.3863
    Epoch 32/100
    188/188 [==============================] - ETA: 0s - loss: 1.2844 - accuracy: 0.5206
    Epoch 00032: val_loss did not improve from 1.61430
    188/188 [==============================] - 37s 195ms/step - loss: 1.2844 - accuracy: 0.5206 - val_loss: 1.6401 - val_accuracy: 0.3949
    Epoch 33/100
    188/188 [==============================] - ETA: 0s - loss: 1.2611 - accuracy: 0.5366
    Epoch 00033: val_loss did not improve from 1.61430
    188/188 [==============================] - 48s 256ms/step - loss: 1.2611 - accuracy: 0.5366 - val_loss: 1.7130 - val_accuracy: 0.3797
    Epoch 34/100
    188/188 [==============================] - ETA: 0s - loss: 1.2517 - accuracy: 0.5411
    Epoch 00034: val_loss did not improve from 1.61430
    188/188 [==============================] - 43s 226ms/step - loss: 1.2517 - accuracy: 0.5411 - val_loss: 1.7434 - val_accuracy: 0.3976
    Epoch 35/100
    188/188 [==============================] - ETA: 0s - loss: 1.2285 - accuracy: 0.5426
    Epoch 00035: val_loss did not improve from 1.61430
    188/188 [==============================] - 39s 209ms/step - loss: 1.2285 - accuracy: 0.5426 - val_loss: 1.7212 - val_accuracy: 0.3910
    Epoch 36/100
    188/188 [==============================] - ETA: 0s - loss: 1.2471 - accuracy: 0.5403
    Epoch 00036: val_loss did not improve from 1.61430
    188/188 [==============================] - 37s 197ms/step - loss: 1.2471 - accuracy: 0.5403 - val_loss: 1.7627 - val_accuracy: 0.3930
    Epoch 37/100
    188/188 [==============================] - ETA: 0s - loss: 1.3133 - accuracy: 0.5165
    Epoch 00037: val_loss did not improve from 1.61430
    188/188 [==============================] - 48s 253ms/step - loss: 1.3133 - accuracy: 0.5165 - val_loss: 1.6753 - val_accuracy: 0.3936
    Epoch 38/100
    188/188 [==============================] - ETA: 0s - loss: 1.2192 - accuracy: 0.5532
    Epoch 00038: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 190ms/step - loss: 1.2192 - accuracy: 0.5532 - val_loss: 1.7208 - val_accuracy: 0.4043
    Epoch 39/100
    188/188 [==============================] - ETA: 0s - loss: 1.1602 - accuracy: 0.5685
    Epoch 00039: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 190ms/step - loss: 1.1602 - accuracy: 0.5685 - val_loss: 1.7728 - val_accuracy: 0.4016
    Epoch 40/100
    188/188 [==============================] - ETA: 0s - loss: 1.1617 - accuracy: 0.5757
    Epoch 00040: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 193ms/step - loss: 1.1617 - accuracy: 0.5757 - val_loss: 1.7637 - val_accuracy: 0.4016
    Epoch 41/100
    188/188 [==============================] - ETA: 0s - loss: 1.1227 - accuracy: 0.5913
    Epoch 00041: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 189ms/step - loss: 1.1227 - accuracy: 0.5913 - val_loss: 1.8768 - val_accuracy: 0.3983
    Epoch 42/100
    188/188 [==============================] - ETA: 0s - loss: 1.1375 - accuracy: 0.5768
    Epoch 00042: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 193ms/step - loss: 1.1375 - accuracy: 0.5768 - val_loss: 2.0369 - val_accuracy: 0.3777
    Epoch 43/100
    188/188 [==============================] - ETA: 0s - loss: 1.1110 - accuracy: 0.5842
    Epoch 00043: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 192ms/step - loss: 1.1110 - accuracy: 0.5842 - val_loss: 1.8281 - val_accuracy: 0.4056
    Epoch 44/100
    188/188 [==============================] - ETA: 0s - loss: 1.0798 - accuracy: 0.5983
    Epoch 00044: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 191ms/step - loss: 1.0798 - accuracy: 0.5983 - val_loss: 1.8059 - val_accuracy: 0.3976
    Epoch 45/100
    188/188 [==============================] - ETA: 0s - loss: 1.2756 - accuracy: 0.5341
    Epoch 00045: val_loss did not improve from 1.61430
    188/188 [==============================] - 37s 194ms/step - loss: 1.2756 - accuracy: 0.5341 - val_loss: 1.7012 - val_accuracy: 0.3943
    Epoch 46/100
    188/188 [==============================] - ETA: 0s - loss: 1.2646 - accuracy: 0.5339
    Epoch 00046: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 190ms/step - loss: 1.2646 - accuracy: 0.5339 - val_loss: 1.7900 - val_accuracy: 0.4009
    Epoch 47/100
    188/188 [==============================] - ETA: 0s - loss: 1.1570 - accuracy: 0.5745
    Epoch 00047: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 190ms/step - loss: 1.1570 - accuracy: 0.5745 - val_loss: 1.8112 - val_accuracy: 0.4003
    Epoch 48/100
    188/188 [==============================] - ETA: 0s - loss: 1.1162 - accuracy: 0.5913
    Epoch 00048: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 190ms/step - loss: 1.1162 - accuracy: 0.5913 - val_loss: 1.8147 - val_accuracy: 0.3949
    Epoch 49/100
    188/188 [==============================] - ETA: 0s - loss: 1.0691 - accuracy: 0.6070
    Epoch 00049: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 190ms/step - loss: 1.0691 - accuracy: 0.6070 - val_loss: 1.8366 - val_accuracy: 0.3963
    Epoch 50/100
    188/188 [==============================] - ETA: 0s - loss: 1.0374 - accuracy: 0.6111
    Epoch 00050: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 191ms/step - loss: 1.0374 - accuracy: 0.6111 - val_loss: 1.8171 - val_accuracy: 0.3936
    Epoch 51/100
    188/188 [==============================] - ETA: 0s - loss: 1.0640 - accuracy: 0.6063
    Epoch 00051: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 189ms/step - loss: 1.0640 - accuracy: 0.6063 - val_loss: 1.8567 - val_accuracy: 0.4102
    Epoch 52/100
    188/188 [==============================] - ETA: 0s - loss: 1.0318 - accuracy: 0.6156
    Epoch 00052: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 190ms/step - loss: 1.0318 - accuracy: 0.6156 - val_loss: 1.9307 - val_accuracy: 0.4009
    Epoch 53/100
    188/188 [==============================] - ETA: 0s - loss: 0.9816 - accuracy: 0.6324
    Epoch 00053: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 190ms/step - loss: 0.9816 - accuracy: 0.6324 - val_loss: 1.9437 - val_accuracy: 0.4109
    Epoch 54/100
    188/188 [==============================] - ETA: 0s - loss: 0.9604 - accuracy: 0.6432
    Epoch 00054: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 190ms/step - loss: 0.9604 - accuracy: 0.6432 - val_loss: 2.0343 - val_accuracy: 0.3910
    Epoch 55/100
    188/188 [==============================] - ETA: 0s - loss: 0.9328 - accuracy: 0.6469
    Epoch 00055: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 192ms/step - loss: 0.9328 - accuracy: 0.6469 - val_loss: 2.1474 - val_accuracy: 0.4056
    Epoch 56/100
    188/188 [==============================] - ETA: 0s - loss: 0.9325 - accuracy: 0.6510
    Epoch 00056: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 190ms/step - loss: 0.9325 - accuracy: 0.6510 - val_loss: 2.0131 - val_accuracy: 0.4023
    Epoch 57/100
    188/188 [==============================] - ETA: 0s - loss: 0.9123 - accuracy: 0.6557
    Epoch 00057: val_loss did not improve from 1.61430
    188/188 [==============================] - 37s 195ms/step - loss: 0.9123 - accuracy: 0.6557 - val_loss: 2.2220 - val_accuracy: 0.4009
    Epoch 58/100
    188/188 [==============================] - ETA: 0s - loss: 0.9007 - accuracy: 0.6617
    Epoch 00058: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 191ms/step - loss: 0.9007 - accuracy: 0.6617 - val_loss: 2.1213 - val_accuracy: 0.4142
    Epoch 59/100
    188/188 [==============================] - ETA: 0s - loss: 0.9012 - accuracy: 0.6620
    Epoch 00059: val_loss did not improve from 1.61430
    188/188 [==============================] - 37s 196ms/step - loss: 0.9012 - accuracy: 0.6620 - val_loss: 1.9507 - val_accuracy: 0.3969
    Epoch 60/100
    188/188 [==============================] - ETA: 0s - loss: 0.8808 - accuracy: 0.6702
    Epoch 00060: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 190ms/step - loss: 0.8808 - accuracy: 0.6702 - val_loss: 2.2631 - val_accuracy: 0.4082
    Epoch 61/100
    188/188 [==============================] - ETA: 0s - loss: 0.9106 - accuracy: 0.6612
    Epoch 00061: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 190ms/step - loss: 0.9106 - accuracy: 0.6612 - val_loss: 2.0874 - val_accuracy: 0.4043
    Epoch 62/100
    188/188 [==============================] - ETA: 0s - loss: 0.9047 - accuracy: 0.6625
    Epoch 00062: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 189ms/step - loss: 0.9047 - accuracy: 0.6625 - val_loss: 2.2813 - val_accuracy: 0.3890
    Epoch 63/100
    188/188 [==============================] - ETA: 0s - loss: 0.8433 - accuracy: 0.6835
    Epoch 00063: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 190ms/step - loss: 0.8433 - accuracy: 0.6835 - val_loss: 2.2580 - val_accuracy: 0.3923
    Epoch 64/100
    188/188 [==============================] - ETA: 0s - loss: 0.8223 - accuracy: 0.6900
    Epoch 00064: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 190ms/step - loss: 0.8223 - accuracy: 0.6900 - val_loss: 2.3449 - val_accuracy: 0.4076
    Epoch 65/100
    188/188 [==============================] - ETA: 0s - loss: 0.8020 - accuracy: 0.6905
    Epoch 00065: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 190ms/step - loss: 0.8020 - accuracy: 0.6905 - val_loss: 2.3353 - val_accuracy: 0.4129
    Epoch 66/100
    188/188 [==============================] - ETA: 0s - loss: 0.8729 - accuracy: 0.6727
    Epoch 00066: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 191ms/step - loss: 0.8729 - accuracy: 0.6727 - val_loss: 2.2367 - val_accuracy: 0.4136
    Epoch 67/100
    188/188 [==============================] - ETA: 0s - loss: 0.8147 - accuracy: 0.6946
    Epoch 00067: val_loss did not improve from 1.61430
    188/188 [==============================] - 37s 194ms/step - loss: 0.8147 - accuracy: 0.6946 - val_loss: 2.2683 - val_accuracy: 0.4003
    Epoch 68/100
    188/188 [==============================] - ETA: 0s - loss: 0.7606 - accuracy: 0.7104
    Epoch 00068: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 189ms/step - loss: 0.7606 - accuracy: 0.7104 - val_loss: 2.4563 - val_accuracy: 0.4176
    Epoch 69/100
    188/188 [==============================] - ETA: 0s - loss: 0.7393 - accuracy: 0.7136
    Epoch 00069: val_loss did not improve from 1.61430
    188/188 [==============================] - 38s 203ms/step - loss: 0.7393 - accuracy: 0.7136 - val_loss: 2.5247 - val_accuracy: 0.4149
    Epoch 70/100
    188/188 [==============================] - ETA: 0s - loss: 0.7413 - accuracy: 0.7176
    Epoch 00070: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 190ms/step - loss: 0.7413 - accuracy: 0.7176 - val_loss: 2.5366 - val_accuracy: 0.4096
    Epoch 71/100
    188/188 [==============================] - ETA: 0s - loss: 0.7208 - accuracy: 0.7282
    Epoch 00071: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 189ms/step - loss: 0.7208 - accuracy: 0.7282 - val_loss: 2.6403 - val_accuracy: 0.4102
    Epoch 72/100
    188/188 [==============================] - ETA: 0s - loss: 0.6976 - accuracy: 0.7372
    Epoch 00072: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 189ms/step - loss: 0.6976 - accuracy: 0.7372 - val_loss: 2.6669 - val_accuracy: 0.4003
    Epoch 73/100
    188/188 [==============================] - ETA: 0s - loss: 0.7195 - accuracy: 0.7272
    Epoch 00073: val_loss did not improve from 1.61430
    188/188 [==============================] - 35s 189ms/step - loss: 0.7195 - accuracy: 0.7272 - val_loss: 2.5599 - val_accuracy: 0.4089
    Epoch 74/100
    188/188 [==============================] - ETA: 0s - loss: 0.6740 - accuracy: 0.7437
    Epoch 00074: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 193ms/step - loss: 0.6740 - accuracy: 0.7437 - val_loss: 2.6475 - val_accuracy: 0.4062
    Epoch 75/100
    188/188 [==============================] - ETA: 0s - loss: 0.9692 - accuracy: 0.6683
    Epoch 00075: val_loss did not improve from 1.61430
    188/188 [==============================] - 37s 195ms/step - loss: 0.9692 - accuracy: 0.6683 - val_loss: 2.3330 - val_accuracy: 0.4036
    Epoch 76/100
    188/188 [==============================] - ETA: 0s - loss: 0.7681 - accuracy: 0.7071
    Epoch 00076: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 189ms/step - loss: 0.7681 - accuracy: 0.7071 - val_loss: 2.6049 - val_accuracy: 0.4082
    Epoch 77/100
    188/188 [==============================] - ETA: 0s - loss: 0.6838 - accuracy: 0.7385
    Epoch 00077: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 191ms/step - loss: 0.6838 - accuracy: 0.7385 - val_loss: 2.6418 - val_accuracy: 0.4096
    Epoch 78/100
    188/188 [==============================] - ETA: 0s - loss: 0.6553 - accuracy: 0.7493
    Epoch 00078: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 190ms/step - loss: 0.6553 - accuracy: 0.7493 - val_loss: 2.9193 - val_accuracy: 0.4142
    Epoch 79/100
    188/188 [==============================] - ETA: 0s - loss: 0.6138 - accuracy: 0.7696
    Epoch 00079: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 190ms/step - loss: 0.6138 - accuracy: 0.7696 - val_loss: 2.9434 - val_accuracy: 0.4129
    Epoch 80/100
    188/188 [==============================] - ETA: 0s - loss: 0.6878 - accuracy: 0.7385
    Epoch 00080: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 190ms/step - loss: 0.6878 - accuracy: 0.7385 - val_loss: 2.7616 - val_accuracy: 0.4089
    Epoch 81/100
    188/188 [==============================] - ETA: 0s - loss: 0.5887 - accuracy: 0.7718
    Epoch 00081: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 190ms/step - loss: 0.5887 - accuracy: 0.7718 - val_loss: 2.9974 - val_accuracy: 0.4089
    Epoch 82/100
    188/188 [==============================] - ETA: 0s - loss: 0.6627 - accuracy: 0.7465
    Epoch 00082: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 190ms/step - loss: 0.6627 - accuracy: 0.7465 - val_loss: 2.8387 - val_accuracy: 0.3983
    Epoch 83/100
    188/188 [==============================] - ETA: 0s - loss: 0.5794 - accuracy: 0.7746
    Epoch 00083: val_loss did not improve from 1.61430
    188/188 [==============================] - 39s 208ms/step - loss: 0.5794 - accuracy: 0.7746 - val_loss: 2.8687 - val_accuracy: 0.4009
    Epoch 84/100
    188/188 [==============================] - ETA: 0s - loss: 0.5617 - accuracy: 0.7809
    Epoch 00084: val_loss did not improve from 1.61430
    188/188 [==============================] - 42s 225ms/step - loss: 0.5617 - accuracy: 0.7809 - val_loss: 3.1431 - val_accuracy: 0.4009
    Epoch 85/100
    188/188 [==============================] - ETA: 0s - loss: 0.5718 - accuracy: 0.7796
    Epoch 00085: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 194ms/step - loss: 0.5718 - accuracy: 0.7796 - val_loss: 3.1206 - val_accuracy: 0.4122
    Epoch 86/100
    188/188 [==============================] - ETA: 0s - loss: 0.6938 - accuracy: 0.7385
    Epoch 00086: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 190ms/step - loss: 0.6938 - accuracy: 0.7385 - val_loss: 2.8433 - val_accuracy: 0.3983
    Epoch 87/100
    188/188 [==============================] - ETA: 0s - loss: 0.6282 - accuracy: 0.7611
    Epoch 00087: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 192ms/step - loss: 0.6282 - accuracy: 0.7611 - val_loss: 2.9247 - val_accuracy: 0.3903
    Epoch 88/100
    188/188 [==============================] - ETA: 0s - loss: 0.5527 - accuracy: 0.7834
    Epoch 00088: val_loss did not improve from 1.61430
    188/188 [==============================] - 40s 211ms/step - loss: 0.5527 - accuracy: 0.7834 - val_loss: 3.2278 - val_accuracy: 0.4043
    Epoch 89/100
    188/188 [==============================] - ETA: 0s - loss: 0.5499 - accuracy: 0.7904
    Epoch 00089: val_loss did not improve from 1.61430
    188/188 [==============================] - 39s 206ms/step - loss: 0.5499 - accuracy: 0.7904 - val_loss: 3.1070 - val_accuracy: 0.4149
    Epoch 90/100
    188/188 [==============================] - ETA: 0s - loss: 0.6586 - accuracy: 0.7615
    Epoch 00090: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 193ms/step - loss: 0.6586 - accuracy: 0.7615 - val_loss: 2.8608 - val_accuracy: 0.3989
    Epoch 91/100
    188/188 [==============================] - ETA: 0s - loss: 0.5201 - accuracy: 0.8024
    Epoch 00091: val_loss did not improve from 1.61430
    188/188 [==============================] - 37s 195ms/step - loss: 0.5201 - accuracy: 0.8024 - val_loss: 3.1098 - val_accuracy: 0.4069
    Epoch 92/100
    188/188 [==============================] - ETA: 0s - loss: 0.4997 - accuracy: 0.8006
    Epoch 00092: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 193ms/step - loss: 0.4997 - accuracy: 0.8006 - val_loss: 3.3906 - val_accuracy: 0.3989
    Epoch 93/100
    188/188 [==============================] - ETA: 0s - loss: 0.4659 - accuracy: 0.8192
    Epoch 00093: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 193ms/step - loss: 0.4659 - accuracy: 0.8192 - val_loss: 3.6647 - val_accuracy: 0.4109
    Epoch 94/100
    188/188 [==============================] - ETA: 0s - loss: 0.5526 - accuracy: 0.7941
    Epoch 00094: val_loss did not improve from 1.61430
    188/188 [==============================] - 37s 198ms/step - loss: 0.5526 - accuracy: 0.7941 - val_loss: 2.9137 - val_accuracy: 0.4182
    Epoch 95/100
    188/188 [==============================] - ETA: 0s - loss: 0.4875 - accuracy: 0.8139
    Epoch 00095: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 191ms/step - loss: 0.4875 - accuracy: 0.8139 - val_loss: 3.4538 - val_accuracy: 0.4189
    Epoch 96/100
    188/188 [==============================] - ETA: 0s - loss: 0.4470 - accuracy: 0.8250
    Epoch 00096: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 192ms/step - loss: 0.4470 - accuracy: 0.8250 - val_loss: 3.5088 - val_accuracy: 0.4082
    Epoch 97/100
    188/188 [==============================] - ETA: 0s - loss: 0.4553 - accuracy: 0.8273
    Epoch 00097: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 192ms/step - loss: 0.4553 - accuracy: 0.8273 - val_loss: 3.5112 - val_accuracy: 0.4182
    Epoch 98/100
    188/188 [==============================] - ETA: 0s - loss: 0.4109 - accuracy: 0.8373
    Epoch 00098: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 193ms/step - loss: 0.4109 - accuracy: 0.8373 - val_loss: 3.6034 - val_accuracy: 0.4069
    Epoch 99/100
    188/188 [==============================] - ETA: 0s - loss: 0.4266 - accuracy: 0.8378
    Epoch 00099: val_loss did not improve from 1.61430
    188/188 [==============================] - 37s 195ms/step - loss: 0.4266 - accuracy: 0.8378 - val_loss: 3.7536 - val_accuracy: 0.4082
    Epoch 100/100
    188/188 [==============================] - ETA: 0s - loss: 0.4021 - accuracy: 0.8430
    Epoch 00100: val_loss did not improve from 1.61430
    188/188 [==============================] - 36s 193ms/step - loss: 0.4021 - accuracy: 0.8430 - val_loss: 3.6920 - val_accuracy: 0.4076



```python
model = load_model("model.h5")
```


```python
def predictions(text):
  clean = re.sub(r'[^ a-z A-Z 0-9]', " ", text)
  test_word = word_tokenize(clean)
  test_word = [w.lower() for w in test_word]
  test_ls = word_tokenizer.texts_to_sequences(test_word)

  if [] in test_ls:
    test_ls = list(filter(None, test_ls))
    
  test_ls = np.array(test_ls).reshape(1, len(test_ls))
  x = padding_doc(test_ls, max_length)

  pred = model.predict(x)
  
  return pred
```


```python
def get_final_output(pred, classes):
  predictions = pred[0]
  classes = np.array(classes)
  ids = np.argsort(-predictions)
  classes = classes[ids]
  predictions = -np.sort(-predictions)
 
  for i in range(pred.shape[1]):
    print("%s has confidence = %s" % (classes[i], (predictions[i])))
  
  return classes[0]
```


```python
text = "I did not help out enough at my thesis team."
pred = predictions(text)
result = get_final_output(pred, unique_label)
print('\nans: {}\n'.format(result))
```

    guilt has confidence = 0.60942477
    shame has confidence = 0.3327749
    anger has confidence = 0.028819958
    sadness has confidence = 0.01870823
    disgust has confidence = 0.005583318
    joy has confidence = 0.002525375
    fear has confidence = 0.0021633662
    
    ans: guilt
    



```python
text = "When someone stole my bike."
pred = predictions(text)
result = get_final_output(pred, unique_label)
print('\nans: {}\n'.format(result))
```

    anger has confidence = 0.4896557
    disgust has confidence = 0.22202596
    guilt has confidence = 0.14396228
    shame has confidence = 0.08689866
    sadness has confidence = 0.04276661
    joy has confidence = 0.009014194
    fear has confidence = 0.0056765988
    
    ans: anger
    



```python
text = "When my girlfriend left me."
pred = predictions(text)
result = get_final_output(pred, unique_label)
print('\nans: {}\n'.format(result))
```

    sadness has confidence = 0.68341756
    joy has confidence = 0.102982506
    anger has confidence = 0.0767743
    disgust has confidence = 0.04667361
    guilt has confidence = 0.043792434
    shame has confidence = 0.029173799
    fear has confidence = 0.01718583
    
    ans: sadness
    



```python
text = "During the Christmas holidays, I met some of my old friends."
pred = predictions(text)
result = get_final_output(pred, unique_label)
print('\nans: {}\n'.format(result))
```

    joy has confidence = 0.63931054
    shame has confidence = 0.123251356
    sadness has confidence = 0.09643768
    guilt has confidence = 0.058223043
    fear has confidence = 0.04968705
    disgust has confidence = 0.021635186
    anger has confidence = 0.011455217
    
    ans: joy
    

