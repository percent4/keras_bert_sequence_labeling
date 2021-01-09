本项目用于将keras_bert_sequnce_labeling生成的h5文件转化为tensorflow/serving支持的pb文件，并实现模型的部署与预测。

### 维护者

-  Jclian91

### 执行顺序

1. 运行change_keras_h5_file_to_pb_models.py文件，会在当前目录下生成example_ner.pb文件。注意，输入、输出文件在脚本的input_model和output_model中设置。
2. 运行get_tf_serving_file.py文件，会在上级目录生成example_ner文件夹，同时可删除当前目录下的example_ner.pb文件。
3. 运行docker命令

```bash
docker run -t --rm -p 8561:8501 -v "$path/example_ner:/models/example_ner" -e MODEL_NAME=example_ner tensorflow/serving:1.14.0
```
其中$path为example_ner文件夹所在的完整路径。

### 模型预测

执行tf_serving_predict.py，输出结果如下：

```
{'entities': [{'end': 34, 'start': 32, 'type': 'LOC', 'word': '香港'},
              {'end': 38, 'start': 36, 'type': 'PER', 'word': '阿毛'},
              {'end': 44, 'start': 39, 'type': 'LOC', 'word': '天安门广场'},
              {'end': 77, 'start': 75, 'type': 'LOC', 'word': '广东'},
              {'end': 103, 'start': 101, 'type': 'LOC', 'word': '北京'}],
 'string': '“看得我热泪盈眶，现场太震撼了。” '
           '2021年1月1日，24岁的香港青年阿毛在天安门广场观看了新年第一次升旗仪式。为了实现这个愿望，他骑着山地自行车从广东出发，风雪兼程，于2020年12月31日下午赶到北京。'}
```

### tensorflow/serving的HTTP接口调用耗时测试报告

测试结果报告

- 普通方式启动docker: 单线程调用1000次，平均每次请求耗时314.5ms
- 启动docker，设置--rest_api_num_threads=160，单线程调用: 调用1000次，平均每次请求耗时379.5ms
- 启动docker，设置--rest_api_num_threads=300，单线程调用: 调用1000次，平均每次请求耗时352.5ms
- 启动docker，设置--rest_api_num_threads=300，10个线程调用: 调用1000次，平均每次请求耗时97.8ms
- 启动docker，设置--rest_api_num_threads=300，20个线程调用: 调用1000次，平均每次请求耗时84.8ms
- 启动docker，设置--enable_batching=true: 单线程调用: 调用1000次，batch_size=10，平均每次请求耗时93.5ms
- 启动docker，设置--rest_api_num_threads=300，--enable_batching=true，10个线程调用: 调用1000次，batch_size=10，平均每次请求耗时64.2ms

测试结果总结：

tensorflow/serving天然地支持并发，在服务端设置rest_api_num_threads和enable_batching，在客户端多线程调用或者批量调用都有助于提高预测效率，多线程+批量调用效率最高。

备注： 以上测试结果与服务端机器性能、客户端机器性能、预测文本数量、预测文本长度等因素有关，因此请求耗时时间仅作为参考。