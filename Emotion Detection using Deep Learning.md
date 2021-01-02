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
from livelossplot.tf_keras import PlotLossesCallback
from livelossplot import PlotLossesKeras
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

    ['fear', 'anger', 'shame', 'sadness', 'joy', 'disgust', 'guilt']



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




    
![png](output_8_1.png)
    



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




    {'fear': 1,
     'anger': 2,
     'shame': 3,
     'sadness': 4,
     'joy': 5,
     'disgust': 6,
     'guilt': 7}




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
                 callbacks = [PlotLossesKeras(), checkpoint])
```


    
![png](output_39_0.png)
    


    accuracy
    	training         	 (min:    0.143, max:    0.936, cur:    0.936)
    	validation       	 (min:    0.186, max:    0.449, cur:    0.427)
    Loss
    	training         	 (min:    0.173, max:    1.947, cur:    0.173)
    	validation       	 (min:    1.556, max:    6.043, cur:    6.043)
    
    Epoch 00100: val_loss did not improve from 1.55581
    188/188 [==============================] - 28s 147ms/step - loss: 0.1726 - accuracy: 0.9356 - val_loss: 6.0430 - val_accuracy: 0.4269


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

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 179, 128)          1150592   
    _________________________________________________________________
    bidirectional_1 (Bidirection (None, 256)               263168    
    _________________________________________________________________
    dense_2 (Dense)              (None, 32)                8224      
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 32)                0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 7)                 231       
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
                 callbacks = [PlotLossesKeras(), checkpoint])
```


    
![png](output_42_0.png)
    


    accuracy
    	training         	 (min:    0.143, max:    0.875, cur:    0.867)
    	validation       	 (min:    0.158, max:    0.437, cur:    0.430)
    Loss
    	training         	 (min:    0.350, max:    1.947, cur:    0.379)
    	validation       	 (min:    1.602, max:    3.759, cur:    3.562)
    
    Epoch 00100: val_loss did not improve from 1.60191
    188/188 [==============================] - 46s 247ms/step - loss: 0.3785 - accuracy: 0.8674 - val_loss: 3.5621 - val_accuracy: 0.4295


## 6. Real-time Prediction


```python
model = load_model("model_lstm.h5")
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
def get_emotion(text):
    pred = predictions(text)
    result = get_final_output(pred, unique_label)
    print('\nans: {}\n'.format(result))
```


```python
get_emotion("I did not help out enough at my thesis team.")
```

    guilt has confidence = 0.49738222
    shame has confidence = 0.36354083
    anger has confidence = 0.05595107
    sadness has confidence = 0.037771154
    fear has confidence = 0.022115987
    disgust has confidence = 0.015596252
    joy has confidence = 0.0076424023
    
    ans: guilt
    



```python
get_emotion("When someone stole my bike.")
```

    anger has confidence = 0.21699908
    fear has confidence = 0.16840756
    guilt has confidence = 0.16005495
    sadness has confidence = 0.15744306
    shame has confidence = 0.14280295
    disgust has confidence = 0.11216525
    joy has confidence = 0.042127196
    
    ans: anger
    



```python
get_emotion("When my girlfriend left me and tell me that I am not fit for her.")
```

    anger has confidence = 0.40398777
    sadness has confidence = 0.19539253
    disgust has confidence = 0.126978
    guilt has confidence = 0.090000965
    joy has confidence = 0.08952445
    shame has confidence = 0.070161685
    fear has confidence = 0.023954567
    
    ans: anger
    



```python
get_emotion("During the Christmas holidays, I met some of my old friends.")
```

    joy has confidence = 0.3547267
    disgust has confidence = 0.2273542
    anger has confidence = 0.15483476
    shame has confidence = 0.105804436
    sadness has confidence = 0.066970326
    guilt has confidence = 0.060981136
    fear has confidence = 0.029328424
    
    ans: joy
    

