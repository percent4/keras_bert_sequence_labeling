# -*- coding: utf-8 -*-
import json
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras_bert import Tokenizer

from util import event_type
from util import MAX_SEQ_LEN, BATCH_SIZE, EPOCH, train_file_path, test_file_path
from load_data import read_data
from model import BertBilstmCRF


# 读取label2id字典
with open("{}_label2id.json".format(event_type), "r", encoding="utf-8") as h:
    label_id_dict = json.loads(h.read())

id_label_dict = {v: k for k, v in label_id_dict.items()}


# 载入数据
# config_path = './chinese_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = './chinese_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = './chinese_L-12_H-768_A-12/vocab.txt'
config_path = './nezha_base/bert_config.json'
checkpoint_path = './nezha_base/bert_model.ckpt'
dict_path = './nezha_base/vocab.txt'


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
    print(input_train_types[0])
    result_train = PreProcessOutputData(result_train)
    # 测试集
    input_test_labels, input_test_types = PreProcessInputData(input_test)
    result_test = PreProcessOutputData(result_test)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=1, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', min_delta=0.0004, patience=2, factor=0.1, min_lr=1e-6,
                                  mode='auto',
                                  verbose=1)
    model = BertBilstmCRF(max_seq_length=MAX_SEQ_LEN, lstm_dim=64).create_model()

    history = model.fit(x=[input_train_labels, input_train_types],
                        y=result_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCH,
                        validation_data=[[input_test_labels, input_test_types], result_test],
                        verbose=1,
                        callbacks=[early_stopping, reduce_lr],
                        shuffle=True
                        )

    # 保存模型
    model.save("{}_nezha_base_ner.h5".format(event_type))

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
