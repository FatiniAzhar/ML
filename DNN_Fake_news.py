# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 22:49:32 2020
#Machine Learning Assignment
@author: Fatini Azhar
"""
#The  source of dataset used was based on https://www.kaggle.com/hassanamin/textdb3

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


#Load Data
df = pd.read_csv('fake_or_real_news.csv')

#Preprocessing
df['text'] = df['text'].apply(lambda x: x.lower())

#Tokennization 
max_features = 2000 #Vocab size


tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(df['text'].values)
X = tokenizer.texts_to_sequences(df['text'].values)


#Padding
max_length = 1000
X = pad_sequences(X,maxlen = max_length, padding = 'post')


#Processing Target
y = df.label
y = pd.get_dummies(df['label']).values

#Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=53)

print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)

# define the model
model = Sequential()
model.add(Embedding(max_features, 24, input_length=max_length))
model.add(Flatten())
model.add(Dense(2, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model.summary())

#Fit the model (Training)
model.fit(X_train, y_train, epochs=50, verbose=0)

# evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %f' % (accuracy*100))





