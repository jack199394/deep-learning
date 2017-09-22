# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 14:40:07 2017

@author: cst-719
"""

#mul-rnn
from __future__ import print_function
import numpy as np
import pandas as pd
import jieba
import csv
import pandas as pd 
import sys
import gensim
import keras

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding,Dropout
from keras.layers import LSTM,SimpleRNN
from keras.datasets import imdb
from keras.layers import Merge


max_features = 20000
#maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

def read_data(data_file):
    comment = pd.read_csv(data_file, encoding='utf-8')
    comlen=len(comment)
    print(comlen)
    cw = lambda x: list(jieba.cut(x)) #定义分词函数
    comment = comment[comment['titlecontent'].notnull()] #仅读取非空评论
    comment['words'] = comment['titlecontent'].apply(cw) #评论分词 
    w = [] #将所有词语整合在一起
    for i in comment['words']:
        w.extend(i)
    dict = pd.DataFrame(pd.Series(w).value_counts()) #统计词的出现次数
    del w
    dict['id']=list(range(1,len(dict)+1))

    get_sent = lambda x: list(dict['id'][x])
    comment['sent'] = comment['words'].apply(get_sent)  
    maxlen = 10
    comment['sent'] = list(sequence.pad_sequences(comment['sent'], maxlen=maxlen))

    train_x = np.array(list(comment['sent']))[:comlen*0.7] #训练集
    train_y = np.array(list(comment['label']))[:comlen*0.7]
    val_x = np.array(list(comment['sent']))[comlen*0.7:comlen*0.8]
    val_y = np.array(list(comment['label']))[comlen*0.7:comlen*0.8]
    test_x = np.array(list(comment['sent']))[comlen*0.8:] #测试集
    test_y = np.array(list(comment['label']))[comlen*0.8:]
    xa = np.array(list(comment['sent'])) #全集
    ya = np.array(list(comment['label']))
    print(train_x)
    print(train_y)
    return train_x, train_y, test_x, test_y,val_x,val_y,len(dict)

data_file1 = "F:\\cs.csv"
data_file2 = "F:\\ct.csv"
data_file3 = "F:\\mr.csv"
data_file4 = "F:\\xg.csv"

data_file21 = "F:\\cs2.csv"
data_file22 = "F:\\ct2.csv"
data_file23 = "F:\\mr2.csv"
data_file24 = "F:\\xg2.csv"

X_train_1,Y_train_1,X_test_1,Y_test_1,val_x_1,val_y_1,dict_len1 = read_data(data_file1)
X_train_2,Y_train_2,X_test_2,Y_test_2,val_x_2,val_y_2,dict_len2 = read_data(data_file2)
X_train_3,Y_train_3,X_test_3,Y_test_3,val_x_3,val_y_3,dict_len3 = read_data(data_file3)
X_train_4,Y_train_4,X_test_4,Y_test_4,val_x_4,val_y_4,dict_len4 = read_data(data_file4)

X_train_21,Y_train_21,X_test_21,Y_test_21,val_x_21,val_y_21,dict_len21 = read_data(data_file21)
X_train_22,Y_train_22,X_test_22,Y_test_22,val_x_22,val_y_22,dict_len22 = read_data(data_file22)
X_train_23,Y_train_23,X_test_23,Y_test_23,val_x_23,val_y_23,dict_len23 = read_data(data_file23)
X_train_24,Y_train_24,X_test_24,Y_test_24,val_x_24,val_y_24,dict_len24 = read_data(data_file24)
train_labels = Y_train_1
test_labels = Y_test_1
val_labels = val_y_1

first_branch = Sequential()
first_branch.add(Embedding(max_features, 256))
first_branch.add(SimpleRNN(128, activation='tanh',return_sequences=True))
first_branch.add(Dropout(0.6))
#first_branch.add(Dense(1))
#first_branch.add(Activation('tanh'))


second_branch = Sequential()
second_branch.add(Embedding(max_features, 256))
second_branch.add(SimpleRNN(128, activation='tanh',return_sequences=True))
second_branch.add(Dropout(0.6))
#second_branch.add(Dense(1))
#second_branch.add(Activation('tanh'))

third_branch = Sequential()
third_branch.add(Embedding(max_features, 256))
third_branch.add(SimpleRNN(128, activation='tanh',return_sequences=True))
third_branch.add(Dropout(0.6))
#third_branch.add(Dense(1))
#third_branch.add(Activation('tanh'))

fourth_branch = Sequential()
fourth_branch.add(Embedding(max_features, 256))
fourth_branch.add(SimpleRNN(128, activation='tanh',return_sequences=True))
fourth_branch.add(Dropout(0.6))
#fourth_branch.add(Dense(1))
#fourth_branch.add(Activation('tanh'))

merged1 = Merge([first_branch, second_branch,third_branch,fourth_branch], mode='concat')

first_branch2 = Sequential()
first_branch2.add(Embedding(max_features, 256))
first_branch2.add(SimpleRNN(128, activation='tanh',return_sequences=True))
first_branch2.add(Dropout(0.6))
#first_branch.add(Dense(1))
#first_branch.add(Activation('tanh'))


second_branch2 = Sequential()
second_branch2.add(Embedding(max_features, 256))
second_branch2.add(SimpleRNN(128, activation='tanh',return_sequences=True))
second_branch2.add(Dropout(0.6))
#second_branch.add(Dense(1))
#second_branch.add(Activation('tanh'))

third_branch2 = Sequential()
third_branch2.add(Embedding(max_features, 256))
third_branch2.add(SimpleRNN(128, activation='tanh',return_sequences=True))
third_branch2.add(Dropout(0.6))
#third_branch.add(Dense(1))
#third_branch.add(Activation('tanh'))

fourth_branch2 = Sequential()
fourth_branch2.add(Embedding(max_features, 256))
fourth_branch2.add(SimpleRNN(128, activation='tanh',return_sequences=True))
fourth_branch2.add(Dropout(0.6))
#fourth_branch.add(Dense(1))
#fourth_branch.add(Activation('tanh'))
merged2 = Merge([first_branch2, second_branch2,third_branch2,fourth_branch2], mode='concat')
merged = Merge([merged1,merged2], mode='concat')

model = Sequential()
model.add(merged)
model.add(SimpleRNN(256,activation='tanh'))

model.add(Dense(1, activation='sigmoid'))
keras.optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer='Adamax',
              loss='binary_crossentropy',
              metrics=['accuracy','recall','f1score'])

model.fit([X_train_1,X_train_2,X_train_3,X_train_4,X_train_21,X_train_22,X_train_23,X_train_24],train_labels,batch_size=batch_size, nb_epoch=10,validation_data=([val_x_1,val_x_2,val_x_3,val_x_4,val_x_21,val_x_22,val_x_23,val_x_24],val_labels))
score, acc,rec,f1 = model.evaluate([X_test_1,X_test_2,X_test_3,X_test_4,X_test_21,X_test_22,X_test_23,X_test_24],test_labels,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
print('Test recall:',rec)
print('Test f1:', f1)