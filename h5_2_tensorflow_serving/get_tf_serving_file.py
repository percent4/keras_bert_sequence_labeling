# -*- coding: utf-8 -*-
# @Time : 2021/1/8 17:32
# @Author : Jclian91
# @File : get_tf_serving_file.py
# @Place : Yangpu, Shanghai
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from keras_bert import get_custom_objects
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy
from keras.models import load_model

custom_objects = get_custom_objects()
for key, value in {'CRF': CRF, 'crf_loss': crf_loss, 'crf_accuracy': crf_accuracy}.items():
    custom_objects[key] = value

export_dir = '../example_ner/1'
graph_pb = './example_ner.pb'
model = load_model('../example_ner.h5', custom_objects=custom_objects)

builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
with tf.gfile.GFile(graph_pb, "rb") as f:
    graph_def = tf.GraphDef()
graph_def.ParseFromString(f.read())


sigs = {}
with tf.Session(graph=tf.Graph()) as sess:
    tf.import_graph_def(graph_def, name="")

    g = tf.get_default_graph()

    sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
        tf.saved_model.signature_def_utils.predict_signature_def(
            inputs={"input_1": g.get_operation_by_name('input_1').outputs[0], "input_2": g.get_operation_by_name('input_2').outputs[0]},
            outputs={"output": g.get_operation_by_name('crf_1/one_hot').outputs[0]}
        )

    builder.add_meta_graph_and_variables(sess,
                                        [tag_constants.SERVING],
                                        signature_def_map = sigs)

    builder.save()
