# -*- coding: utf-8 -*-
# @Time : 2021/1/11 10:10
# @Author : Jclian91
# @File : model_generator_train.py
# @Place : Yangpu, Shanghai
import json
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras_bert import Tokenizer

from util import event_type, BASE_MODEL_DIR
from util import MAX_SEQ_LEN, BATCH_SIZE, EPOCH, train_file_path, test_file_path
from load_data import read_data
from model import BertBilstmCRF


# 读取label2id字典
with open("{}_label2id.json".format(event_type), "r", encoding="utf-8") as h:
    label_id_dict = json.loads(h.read())

id_label_dict = {v: k for k, v in label_id_dict.items()}


# 载入数据
config_path = './{}/bert_config.json'.format(BASE_MODEL_DIR)
checkpoint_path = './{}/bert_model.ckpt'.format(BASE_MODEL_DIR)
dict_path = './{}/vocab.txt'.format(BASE_MODEL_DIR)


token_dict = {}
with open(dict_path, 'r', encoding='utf-8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            else:
                R.append('[UNK]')   # 剩余的字符是[UNK]
        return R


tokenizer = OurTokenizer(token_dict)


# 预处理输入数据
def PreProcessInputData(text):
    word_labels = []
    seq_types = []
    for sequence in text:
        code = tokenizer.encode(first=sequence, max_len=MAX_SEQ_LEN)
        word_labels.append(code[0])
        seq_types.append(code[1])
    return word_labels, seq_types


# 预处理结果数据
def PreProcessOutputData(text):
    tags = []
    for line in text:
        tag = [0]
        for item in line:
            tag.append(int(label_id_dict[item.strip()]))
        tag.append(0)
        tags.append(tag)

    pad_tags = pad_sequences(tags, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")
    result_tags = np.expand_dims(pad_tags, 2)
    return result_tags


class DataGenerator:
    def __init__(self, token_ids, segment_ids, y_labels, batch_size=BATCH_SIZE):
        self.token_ids = token_ids
        self.segment_ids = segment_ids
        self.y_labels = y_labels
        self.batch_size = batch_size
        self.steps = len(self.token_ids) // self.batch_size
        if len(self.token_ids) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.token_ids)))
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                X1.append(self.token_ids[i])
                X2.append(self.segment_ids[i])
                Y.append(self.y_labels[i])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    yield [np.array(X1), np.array(X2)], np.array(Y)
                    [X1, X2, Y] = [], [], []


if __name__ == '__main__':
    # 读取训练集和测试集数据
    input_train, result_train = read_data(train_file_path)
    input_test, result_test = read_data(test_file_path)
    for sent, tag in zip(input_train[:10], result_train[:10]):
        print(sent, tag)
    for sent, tag in zip(input_test[:10], result_test[:10]):
        print(sent, tag)

    # 训练集
    input_train_labels, input_train_types = PreProcessInputData(input_train)
    result_train = PreProcessOutputData(result_train)
    # 测试集
    input_test_labels, input_test_types = PreProcessInputData(input_test)
    result_test = PreProcessOutputData(result_test)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=1, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', min_delta=0.0004, patience=3, factor=0.1, min_lr=1e-6,
                                  mode='auto',
                                  verbose=1)
    model = BertBilstmCRF(max_seq_length=MAX_SEQ_LEN, lstm_dim=100).create_model()

    train_D = DataGenerator(input_train_labels, input_train_types, result_train)
    test_D = DataGenerator(input_test_labels, input_test_types, result_test)

    history = model.fit_generator(
                        train_D.__iter__(),
                        steps_per_epoch=len(train_D),
                        epochs=EPOCH,
                        validation_data=test_D.__iter__(),
                        validation_steps=len(test_D),
                        verbose=1,
                        shuffle=True,
                        callbacks=[early_stopping, reduce_lr]
                        )

    # 保存模型
    model.save("{}_{}_ner.h5".format(BASE_MODEL_DIR, event_type))

    # 绘制loss和acc图像
    plt.subplot(2, 1, 1)
    epochs = len(history.history['loss'])
    plt.plot(range(epochs), history.history['loss'], label='loss')
    plt.plot(range(epochs), history.history['val_loss'], label='val_loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    epochs = len(history.history['crf_accuracy'])
    plt.plot(range(epochs), history.history['crf_accuracy'], label='crf_accuracy')
    plt.plot(range(epochs), history.history['val_crf_accuracy'], label='val_crf_accuracy')
    plt.legend()
    plt.savefig("%s_loss_acc.png" % event_type)
