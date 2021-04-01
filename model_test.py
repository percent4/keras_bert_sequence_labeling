# -*- coding: utf-8 -*-
# @Time : 2021/3/31 11:27
# @Author : Jclian91
# @File : model_test.py
# @Place : Yangpu, Shanghai
import json
from keras.layers import *
from keras.models import Model
from keras_bert import load_trained_model_from_checkpoint, get_model, load_model_weights_from_checkpoint
from keras_contrib.layers import CRF


from util import BASE_MODEL_DIR

model_path = "./{}/".format(BASE_MODEL_DIR)
seq_len = 128
with open(model_path + "bert_config.json", 'r') as reader:
    config = json.loads(reader.read())
if seq_len is not None:
    config['max_position_embeddings'] = seq_len = min(seq_len, config['max_position_embeddings'])

bert = get_model(
                token_num=config['vocab_size'],
                pos_num=config['max_position_embeddings'],
                seq_len=seq_len,
                embed_dim=config['hidden_size'],
                transformer_num=config['num_hidden_layers'],
                head_num=config['num_attention_heads'],
                feed_forward_dim=config['intermediate_size'],
                feed_forward_activation=config['hidden_act'],
                training=None,
                trainable=True,
                output_layer_num=1,
                )

inputs, outputs = bert
print(type(bert), type(outputs))
load_model_weights_from_checkpoint(outputs, config, model_path + "bert_model.ckpt")

x1 = Input(shape=(None,))
x2 = Input(shape=(None,))
bert_out = outputs.output([x1, x2])
lstm_out = Bidirectional(LSTM(64,
                              return_sequences=True,
                              dropout=0.2,
                              recurrent_dropout=0.2))(bert_out)
crf_out = CRF(8, sparse_target=True)(lstm_out)
model = Model(inputs=[x1, x2], outputs=crf_out)


model.summary()
