本项目采用Keras和Keras-bert实现序列标注。

### 维护者

- jclian91

### 数据集

1. 人民日报命名实体识别数据集（example.train 28046条数据和example.test 4636条数据），共3种标签：地点（LOC）, 人名（PER）, 组织机构（ORG）
2. 时间识别数据集（time.train 1700条数据和time.test 300条数据），共1种标签：TIME
3. CLUENER细粒度实体识别数据集（cluener.train 10748条数据和cluener.test 1343条数据），共10种标签：地址（address），书名（book），公司（company），游戏（game），政府（goverment），电影（movie），姓名（name），组织机构（organization），职位（position），景点（scene）

### 模型结构

```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_3 (InputLayer)            (None, None)         0                                            
__________________________________________________________________________________________________
input_4 (InputLayer)            (None, None)         0                                            
__________________________________________________________________________________________________
model_5 (Model)                 multiple             101382144   input_3[0][0]                    
                                                                 input_4[0][0]                    
__________________________________________________________________________________________________
bidirectional_2 (Bidirectional) (None, None, 200)    695200      model_5[1][0]                    
__________________________________________________________________________________________________
crf_2 (CRF)                     (None, None, 7)      1470        bidirectional_2[0][0]            
==================================================================================================
Total params: 102,078,814
Trainable params: 102,078,814
Non-trainable params: 0
```

### 模型效果

- 人民日报命名实体识别数据集

模型参数：MAX_SEQ_LEN=128, BATCH_SIZE=32, EPOCH=10

运行model_evaluate.py,模型评估结果如下：

```
             precision  recall    f1-score   support
      LOC     0.9330    0.8986    0.9155      3658
      ORG     0.8881    0.8902    0.8891      2185
      PER     0.9692    0.9469    0.9579      1864

micro avg     0.9287    0.9079    0.9182      7707
macro avg     0.9291    0.9079    0.9183      7707
```

- 时间识别数据集

模型参数：MAX_SEQ_LEN=256, BATCH_SIZE=8, EPOCH=10

运行model_evaluate.py,模型评估结果如下：

```
            precision   recall  f1-score   support
     TIME     0.8428    0.8753    0.8587       441

micro avg     0.8428    0.8753    0.8587       441
macro avg     0.8428    0.8753    0.8587       441
```

- CLUENER细粒度实体识别数据集

模型参数：MAX_SEQ_LEN=128, BATCH_SIZE=32, EPOCH=10

运行model_evaluate.py,模型评估结果如下：

```
                precision  recall    f1-score   support
        name     0.8476    0.8758    0.8615       451
       scene     0.6569    0.6734    0.6650       199
    position     0.7455    0.7788    0.7618       425
organization     0.7377    0.7849    0.7606       344
        game     0.7423    0.8432    0.7896       287
     address     0.6070    0.6236    0.6152       364
     company     0.7264    0.7978    0.7604       366
       movie     0.7687    0.7533    0.7609       150
  government     0.7860    0.8279    0.8064       244
        book     0.8041    0.7829    0.7933       152

   micro avg     0.7419    0.7797    0.7603      2982
   macro avg     0.7420    0.7797    0.7601      2982
```

### 模型预测示例

- 人民日报命名实体识别数据集

运行model_predict.py，对新文本进行预测，结果如下：

```
{'entities': [{'end': 17, 'start': 16, 'type': 'LOC', 'word': '欧'},
              {'end': 50, 'start': 48, 'type': 'LOC', 'word': '英国'},
              {'end': 63, 'start': 62, 'type': 'LOC', 'word': '欧'},
              {'end': 72, 'start': 69, 'type': 'PER', 'word': '卡梅伦'},
              {'end': 78, 'start': 73, 'type': 'PER', 'word': '特雷莎·梅'},
              {'end': 86, 'start': 85, 'type': 'LOC', 'word': '欧'},
              {'end': 102, 'start': 95, 'type': 'PER', 'word': '鲍里斯·约翰逊'}],
 'string': '当2016年6月24日凌晨，“脱欧”公投的最后一张选票计算完毕，占投票总数52%的支持选票最终让英国开始了一段长达4年的“脱欧”进程，其间卡梅伦、特雷莎·梅相继离任，“脱欧”最终在第三位首相鲍里斯·约翰逊任内完成。'}
```

```
{'entities': [{'end': 6, 'start': 0, 'type': 'ORG', 'word': '台湾“立法院'},
              {'end': 30, 'start': 29, 'type': 'LOC', 'word': '台'},
              {'end': 38, 'start': 35, 'type': 'PER', 'word': '蔡英文'},
              {'end': 66, 'start': 64, 'type': 'LOC', 'word': '台湾'}],
 'string': '台湾“立法院”“莱猪（含莱克多巴胺的猪肉）”表决大战落幕，台当局领导人蔡英文24日晚在脸书发文宣称，“开放市场的决定，将会是未来台湾国际经贸走向世界的关键决定”。'}
```

```
{'entities': [{'end': 9, 'start': 7, 'type': 'LOC', 'word': '印度'},
              {'end': 14, 'start': 12, 'type': 'LOC', 'word': '南海'},
              {'end': 27, 'start': 25, 'type': 'LOC', 'word': '印度'},
              {'end': 30, 'start': 28, 'type': 'LOC', 'word': '越南'},
              {'end': 45, 'start': 43, 'type': 'LOC', 'word': '印度'},
              {'end': 49, 'start': 47, 'type': 'PER', 'word': '莫迪'},
              {'end': 53, 'start': 51, 'type': 'LOC', 'word': '南海'},
              {'end': 90, 'start': 88, 'type': 'LOC', 'word': '南海'}],
 'string': '最近一段时间，印度政府在南海问题上接连发声。在近期印度、越南两国举行的线上总理峰会上，印度总理莫迪声称南海行为准则“不应损害该地区其他国家或第三方的利益”，两国总理还强调了所谓南海“航行自由”的重要性。'}
```

- 时间识别数据集

运行model_predict.py，对新文本进行预测，结果如下：

```
{'entities': [{'end': 8, 'start': 0, 'type': 'TIME', 'word': '去年11月30日'}],
 'string': '去年11月30日，李先生来到茶店子东街一家银行取钱，准备购买家具。输入密码后，'}
```

```
{'entities': [{'end': 19, 'start': 10, 'type': 'TIME', 'word': '上世纪80年代之前'},
              {'end': 24, 'start': 20, 'type': 'TIME', 'word': '去年9月'},
              {'end': 47, 'start': 45, 'type': 'TIME', 'word': '3年'}],
 'string': '苏北大量农村住房建于上世纪80年代之前。去年9月，江苏省决定全面改善苏北农民住房条件，计划3年内改善30万户，作为决胜全面建成小康社会补短板的重要举措。'}
```

```
{'entities': [{'end': 8, 'start': 6, 'type': 'TIME', 'word': '两天'},
              {'end': 23, 'start': 21, 'type': 'TIME', 'word': '昨天'},
              {'end': 61, 'start': 56, 'type': 'TIME', 'word': '8月10日'},
              {'end': 69, 'start': 64, 'type': 'TIME', 'word': '2016年'}],
 'string': '经过工作人员两天的反复验证、严密测算，记者昨天从上海中心大厦得到确认：被誉为上海中心大厦“定楼神器”的阻尼器，在8月10日出现自2016年正式启用以来的最大摆幅。'}
```

- CLUENER细粒度实体识别数据集

运行model_predict.py，对新文本进行预测，结果如下：

```
{'entities': [{'end': 5, 'start': 0, 'type': 'organization', 'word': '四川敦煌学'},
              {'end': 13, 'start': 11, 'type': 'scene', 'word': '丹棱'},
              {'end': 44, 'start': 41, 'type': 'name', 'word': '胡文和'}],
 'string': '四川敦煌学”。近年来，丹棱县等地一些不知名的石窟迎来了海内外的游客，他们随身携带着胡文和的著作。'}
```

```
{'entities': [{'end': 19, 'start': 14, 'type': 'address', 'word': '茶店子东街'}],
 'string': '去年11月30日，李先生来到茶店子东街一家银行取钱，准备购买家具。输入密码后，'}
```

```
{'entities': [{'end': 3, 'start': 0, 'type': 'name', 'word': '罗伯茨'},
              {'end': 10, 'start': 4, 'type': 'movie', 'word': '《逃跑新娘》'},
              {'end': 23, 'start': 16, 'type': 'movie', 'word': '《理发师佐翰》'},
              {'end': 38, 'start': 32, 'type': 'name', 'word': '亚当·桑德勒'}],
 'string': '罗伯茨的《逃跑新娘》不相伯仲；而《理发师佐翰》让近年来顺风顺水的亚当·桑德勒首尝冲过1亿＄'}
```

### BERT-base与NEZHA-base模型对比

- 模型效果对比

example数据集

```
           precision    recall  f1-score   support

      PER     0.9584    0.9405    0.9494      1864
      ORG     0.8628    0.8810    0.8718      2185
      LOC     0.9311    0.8934    0.9118      3658

micro avg     0.9176    0.9013    0.9093      7707
macro avg     0.9183    0.9013    0.9096      7707
```

time数据集

```

```

cluener数据集

```
              precision    recall  f1-score   support

  government     0.7168    0.8402    0.7736       244
     company     0.7526    0.7814    0.7668       366
     address     0.5235    0.6126    0.5646       364
organization     0.7207    0.7878    0.7528       344
    position     0.7348    0.7953    0.7638       425
        name     0.8274    0.8714    0.8488       451
        game     0.7163    0.8711    0.7862       287
       scene     0.6193    0.6784    0.6475       199
       movie     0.7708    0.7400    0.7551       150
        book     0.7169    0.7829    0.7484       152

   micro avg     0.7107    0.7817    0.7445      2982
   macro avg     0.7135    0.7817    0.7454      2982
```

- 预测速度对比

|模型名称|预测时间|
|---|---|
|BERT-Base|800-900ms|
|Nezha-Base|800-900ms|

结论：模型预测速度并没有显著提升。

### 代码说明

0. 将BERT中文预训练模型chinese_L-12_H-768_A-12放在chinese_L-12_H-768_A-12文件夹下
1. 运行load_data.py，生成类别标签，注意O标签为0;
2. 所需Python第三方模块参考requirements.txt文档
3. 自己需要分类的数据按照data/example.train和data/example.test的格式准备好
4. 调整模型参数，运行model_train.py进行模型训练
5. 运行model_evaluate.py进行模型评估
6. 运行model_predict.py对新文本进行预测