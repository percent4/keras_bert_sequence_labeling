# -*- coding: utf-8 -*-
# @Time : 2021/1/8 18:38
# @Author : Jclian91
# @File : tf_serving_predict.py
# @Place : Yangpu, Shanghai
import json
import requests
import numpy as np
from pprint import pprint
from keras_bert import Tokenizer

from util import event_type, MAX_SEQ_LEN

# 读取label2id字典
with open("../{}_label2id.json".format(event_type), "r", encoding="utf-8") as h:
    label_id_dict = json.loads(h.read())

id_label_dict = {v: k for k, v in label_id_dict.items()}


# 载入数据
dict_path = '../chinese_L-12_H-768_A-12/vocab.txt'
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
                R.append('[UNK]')
        return R


# 将BIO序列转化为JSON格式
def bio_to_json(string, tags):
    item = {"string": string, "entities": []}
    entity_name = ""
    entity_start = 0
    iCount = 0
    entity_tag = ""

    for c_idx in range(min(len(string), len(tags))):
        c, tag = string[c_idx], tags[c_idx]
        if c_idx < len(tags)-1:
            tag_next = tags[c_idx+1]
        else:
            tag_next = ''

        if tag[0] == 'B':
            entity_tag = tag[2:]
            entity_name = c
            entity_start = iCount
            if tag_next[2:] != entity_tag:
                item["entities"].append({"word": c, "start": iCount, "end": iCount + 1, "type": tag[2:]})
        elif tag[0] == "I":
            if tag[2:] != tags[c_idx-1][2:] or tags[c_idx-1][2:] == 'O':
                tags[c_idx] = 'O'
                pass
            else:
                entity_name = entity_name + c
                if tag_next[2:] != entity_tag:
                    item["entities"].append({"word": entity_name, "start": entity_start, "end": iCount + 1, "type": entity_tag})
                    entity_name = ''
        iCount += 1
    return item


tokenizer = OurTokenizer(token_dict)

import time
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
start_time = time.time()
test_num = 100
batch_size = 10
# 测试HTTP响应时间
# for i in range(test_num//batch_size):
#     tensor = {"instances": []}
#     for i in range(batch_size):
#         sentence = "“看得我热泪盈眶，现场太震撼了。” 2021年1月1日，24岁的香港青年阿毛在天安门广场观看了新年第一次升旗仪式。为了实现这个愿望，他骑着山地自行车从广东出发，风雪兼程，于2020年12月31日下午赶到北京。"
#         token_ids, segment_is = tokenizer.encode(sentence, max_len=MAX_SEQ_LEN)
#         tensor["instances"].append({"input_1": token_ids, "input_2": segment_is})
#
#     url = "http://192.168.1.193:8561/v1/models/example_ner:predict"
#     req = requests.post(url, json=tensor)
#     if req.status_code == 200:
#         for j in range(len(req.json()['predictions'])):
#             t = np.asarray(req.json()['predictions'][j]).argmax(axis=1)
#             tags = [id_label_dict[_] for _ in t]
#             print(j, bio_to_json(sentence, tags[1:-1]))


def get_predict(i, sentence_list):
    tensor = {"instances": []}
    for sentence in sentence_list:
        token_ids, segment_is = tokenizer.encode(sentence, max_len=MAX_SEQ_LEN)
        tensor["instances"].append({"input_1": token_ids, "input_2": segment_is})

    url = "http://192.168.1.193:8561/v1/models/example_ner:predict"
    req = requests.post(url, json=tensor)
    if req.status_code == 200:
        for j in range(len(req.json()['predictions'])):
            t = np.asarray(req.json()['predictions'][j]).argmax(axis=1)
            tags = [id_label_dict[_] for _ in t]
            print("predict {} sample, batch no {}, result: {}".format(i, j, bio_to_json(sentence_list[j], tags[1:-1])))

sentence_list = ["“看得我热泪盈眶，现场太震撼了。” 2021年1月1日，24岁的香港青年阿毛在天安门广场观看了新年第一次升旗仪式。为了实现这个愿望，他骑着山地自行车从广东出发，风雪兼程，于2020年12月31日下午赶到北京。"]*batch_size
# 利用多线程调用接口
executor = ThreadPoolExecutor(max_workers=10)  # 可以自己调整max_workers,即线程的个数
# submit()的参数： 第一个为函数， 之后为该函数的传入参数，允许有多个
future_tasks = [executor.submit(get_predict, i, sentence_list) for i in range(test_num//batch_size)]
# 等待所有的线程完成，才进入后续的执行
wait(future_tasks, return_when=ALL_COMPLETED)

end_time = time.time()
print("avg cost time: {}".format((end_time-start_time)/test_num))