import os
from google.colab import drive
drive.mount('/content/drive')

path = "/content/drive/My Drive"

os.chdir(path)
os.listdir(path)

! pip install keras-bert

import pandas as pd
import codecs
import keras.backend as K
import os
import json
import keras
from keras.callbacks import *
import codecs
import numpy as np
from keras.models import Model
import tensorflow as tf
import gc 
from random import choice
from keras.layers import *
from sklearn.model_selection import StratifiedKFold
import re
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.optimizers import Adam

train_data = []
with codecs.open("./o2o/train.csv", 'r', 'utf-8') as f:
    for line in f.readlines():
        '''
            读取train数据，而后转为符合格式的dataframe
        '''
        label, comment = line.strip().split('\t')
        train_data.append([label, comment])
test_data = pd.read_csv('./o2o/test_new.csv')
sample = pd.read_csv('./o2o/sample.csv')

train_data = pd.DataFrame(train_data[1:], columns=train_data[0])
train_data = train_data[['comment', 'label']]

out_path = './'

# 预训练模型
bert_paths = [
    # {
    #     'config_path' : '../input/chinese-roberta/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json',
    #     'checkpoint_path' : '../input/chinese-roberta/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt',
    #     'dict_path' : '../input/chinese-roberta/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'
    # },
    # {
    #     'config_path' : '../input/chinese-bert/bert_config.json',
    #     'checkpoint_path' : '../input/chinese-bert/bert_model.ckpt',
    #     'dict_path' : '../input/chinese-bert/vocab.txt'
    # },
    # {
    #     'config_path' : '../input/chinese-wwm-ext-l12-h768-a12/bert_config.json',
    #     'checkpoint_path' : '../input/chinese-wwm-ext-l12-h768-a12/bert_model.ckpt',
    #     'dict_path' : '../input/chinese-wwm-ext-l12-h768-a12/vocab.txt'
    # },
    {
        'config_path' : './chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json',
        'checkpoint_path' : './chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt',
        'dict_path' : './chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'
    },
    {
        'config_path' : './publish/bert_config.json',
        'checkpoint_path' : './publish/bert_model.ckpt',
        'dict_path' : './publish/vocab.txt'
    },
    {
        'config_path' : '../input/chinese-wwm-ext-l12-h768-a12/bert_config.json',
        'checkpoint_path' : '../input/chinese-wwm-ext-l12-h768-a12/bert_model.ckpt',
        'dict_path' : '../input/chinese-wwm-ext-l12-h768-a12/vocab.txt'
    }
]



maxlen = 170
batch_size = 8
num_epochs = 6
learning_rate = 2e-5
nfold = 5
SEED = 2021

token_dict = {}


def genDict(dict_path):
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            '''
                生成预训练字典
            '''
            token = line.strip()
            token_dict[token] = len(token_dict)

genDict(bert_paths[0]['dict_path'])





'''
    生成分词器
'''
class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            '''
                帮助正确处理空格与不在字典中的字
            '''
            if c in self._token_dict:
                R.append(c)
                '''
                    帮助正确处理空格
                '''
            elif self._is_space(c):
                R.append('[unused1]')
            else:
                '''
                    帮助正确处理不在字典中的字
                '''
                R.append('[UNK]')
        return R

def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    '''
        取最长长度
    '''
    return np.array([np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X])

tokenizer = OurTokenizer(token_dict)



class data_generator:
    def __init__(self, data, tokenizer, batch_size=batch_size, shuffle=True):
        self.data = data
        self.steps = len(self.data) // batch_size
        if len(self.data) % batch_size != 0:
            self.steps += 1
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        '''
            是否 随机打乱
        '''
        self.shuffle = shuffle
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))

            X1, X2, Y = [], [], []
            for i in idxs:
                '''
                    用分词器进行分词
                '''
                d = self.data[i]
                text = d[0][:maxlen]
                x1, x2 = self.tokenizer.encode(first=text)
                y = d[1]
                Y.append([y])
                X1.append(x1)
                X2.append(x2)
                '''
                    padding sequence
                '''
                if len(X1) == self.batch_size or i == idxs[-1]:
                    Y = np.array(Y)
                    '''
                        对x1，x2 进行padding
                    '''
                    X2 = seq_padding(X2)
                    X1 = seq_padding(X1)
                    '''
                        返回数据，传给 model
                    '''
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []

    def __len__(self):
        return self.steps


'''
    使用 tf1 配合kerasbert
'''
tf.compat.v1.disable_eager_execution()
import warnings
warnings.filterwarnings('ignore')

'''
    ------------------------ 构建 bert 模型 ---------------------------
'''
def build_bert(config_path, checkpoint_path):
    bert_model = load_trained_model_from_checkpoint(
        config_path,
        checkpoint_path,
        seq_len=None)

    '''
        对于bert层继续训练
    '''
    for l in bert_model.layers:
        l.trainable = True

    '''
        构建输入
    '''
    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    '''
        取cls向量
    '''
    x = Lambda(lambda x: x[:, 0])(x)
    p = Dense(1, activation='sigmoid')(x)

    model = Model([x1_in, x2_in], p)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate),# 用足够小的学习率
                  metrics=['accuracy'])
    model.summary()
    return model

def train_model(train_D, valid_D, test_D, key, index, directory):
    model = build_bert(bert_paths[0]['config_path'], bert_paths[0]['checkpoint_path'])

    # 处理好模型的路径
    model.load_weights(directory + 'model' + str(index) + '.hdf5')

    '''
        配置 model 训练时的 回调函数
    '''
    '''
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    plateau = ReduceLROnPlateau(monitor="val_loss", verbose=1, mode='min', factor=0.2, patience=2)
    checkpoint = ModelCheckpoint(out_path + 'model_' + str(index) + '_' + str(key) + "_" + str(i) + '.hdf5', monitor='val_loss',
                                    verbose=2, save_best_only=True, mode='min', save_weights_only=True)
    '''
    '''
        开始训练
    '''
    '''
    model.fit_generator(
        train_D.__iter__(),
        validation_steps=len(valid_D),
        epochs = num_epochs,
        steps_per_epoch=len(train_D),
        callbacks=[early_stopping, plateau, checkpoint],
        validation_data=valid_D.__iter__(),
    )
    '''
    # model.predict_generator(valid_D.__iter__(), steps=len(valid_D), verbose=1)
    pred =  model.predict_generator(test_D.__iter__(), steps=len(test_D), verbose=1)
    '''
        省点内存
    '''
    del model
    gc.collect()
    K.clear_session()
    return pred


def run_cv(nfold, data, data_test):
    test_model_pred_1 = np.zeros((len(data_test), 1))
    test_model_pred_2 = np.zeros((len(data_test), 1))
    test_model_pred_3 = np.zeros((len(data_test), 1))
    '''
        进行随机切分
    '''
    skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=SEED).split(data[:, 0], data[:, 1])

    for i, (train_fold, valid_fold) in enumerate(skf):
        print('第%s折开始训练' % (i + 1))
        X_train, X_valid, = data[train_fold, :], data[valid_fold, :]

        '''
            分别对三个模型 进行 三次训练
        '''
        train_D = data_generator(X_train, tokenizer=tokenizer)
        valid_D = data_generator(X_valid, tokenizer=tokenizer)
        test_D = data_generator(data_test, tokenizer=tokenizer, shuffle=False)
        test_model_pred_1 += train_model(train_D, valid_D, test_D, 0, i, './one/') / nfold

        train_D = data_generator(X_train, tokenizer=tokenizer)
        valid_D = data_generator(X_valid, tokenizer=tokenizer)
        test_D = data_generator(data_test, tokenizer=tokenizer, shuffle=False)
        test_model_pred_2 += train_model(train_D, valid_D, test_D, 1, i, './two/') / nfold

        train_D = data_generator(X_train, tokenizer=tokenizer)
        valid_D = data_generator(X_valid, tokenizer=tokenizer)
        test_D = data_generator(data_test, tokenizer=tokenizer, shuffle=False)
        test_model_pred_3 += train_model(train_D, valid_D, test_D, 2, i, './three/') / nfold

    return [test_model_pred_1, test_model_pred_2, test_model_pred_3]

'''
    --------------------------  run  ----------------------------
'''

DATA_LIST = []
for data_row in train_data.iloc[:].itertuples():
    DATA_LIST.append((data_row.comment, data_row.label))

DATA_LIST_TEST = []
for data_row in test_data.iloc[:].itertuples():
    DATA_LIST_TEST.append((data_row.comment, 0))
'''
    生成 array 类型的数据
'''
DATA_LIST = np.array(DATA_LIST)
DATA_LIST_TEST = np.array(DATA_LIST_TEST)



test_model_preds = run_cv(nfold, DATA_LIST, DATA_LIST_TEST)

'''
    对各模型预测概率进行处理
'''
def getRes(predList, threshold):
    re = []
    for i in predList:
        if i[0] > threshold:
            re.append(1)
        else:
            re.append(0)
    return re
def getResForList(predList, threshold):
    re = []
    for i in predList:
        if i > threshold:
            re.append(1)
        else:
            re.append(0)
    return re

re_1 = sample.copy()
re_1['label'] = re_1['label'].apply(lambda x: 0)
re_1['label'] = getRes(test_model_preds[0].tolist(), 0.627)

re_2 = sample.copy()
re_2['label'] = re_2['label'].apply(lambda x: 0)
re_2['label'] = getRes(test_model_preds[1].tolist(), 0.6)

re_3 = sample.copy()
re_3['label'] = re_3['label'].apply(lambda x: 0)
re_3['label'] = getRes(test_model_preds[2].tolist(), 0.6)

label = []
for i in range(len(re_1['label'])):
    label.append((re_1['label'].iloc[i] + re_2['label'].iloc[i] + re_3['label'].iloc[i]) / 3)
re_4 = sample.copy()
re_4['label'] = re_4['label'].apply(lambda x: 0)
re_4['label'] = getResForList(label, 0.6)

np.save('./test_model_preds.npy', test_model_preds)


'''
    对特殊的敏感词进行特殊处理
'''
a = set()
r = test_data
r['label'] = re_4['label']
r0 = r[r.label == 0]
key_word = ['蚊子', '老鼠', '苍蝇', '酸臭', '骚味', '苍蝇', '虫', '臭', '想吐', '太硬']
for i in key_word:
    for j in r0[r0['comment'].str.contains(i)].index.to_list():
        a.add(j)

for i in a:
    re_4.loc[i, 'label'] = [1]
re_4.to_csv('./result.csv', index=None)