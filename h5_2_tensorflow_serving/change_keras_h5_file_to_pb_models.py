# -*- coding: utf-8 -*-
"""
Copyright (c) 2018, by the Authors: Amir H. Abdi
This script is freely available under the MIT Public License.
Please see the License file in the root for details.
The following code snippet will convert the keras model files
to the freezed .pb tensorflow weight file. The resultant TensorFlow model
holds both the model architecture and its associated weights.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from pathlib import Path
from absl import app
from absl import flags
from absl import logging
import keras
from keras import backend as K
from keras.models import model_from_json

from keras_bert import get_custom_objects
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy

custom_objects = get_custom_objects()
for key, value in {'CRF': CRF, 'crf_loss': crf_loss, 'crf_accuracy': crf_accuracy}.items():
    custom_objects[key] = value

K.set_learning_phase(0)
FLAGS = flags.FLAGS

flags.DEFINE_string('input_model', "../example_ner.h5", 'Path to the input model.')
flags.DEFINE_string('input_model_json', None, 'Path to the input model '
                                              'architecture in json format.')
flags.DEFINE_string('output_model', "./example_ner.pb", 'Path where the converted model will '
                                          'be stored.')
flags.DEFINE_boolean('save_graph_def', False,
                     'Whether to save the graphdef.pbtxt file which contains '
                     'the graph definition in ASCII format.')
flags.DEFINE_string('output_nodes_prefix', None,
                    'If set, the output nodes will be renamed to '
                    '`output_nodes_prefix`+i, where `i` will numerate the '
                    'number of of output nodes of the network.')
flags.DEFINE_boolean('quantize', False,
                     'If set, the resultant TensorFlow graph weights will be '
                     'converted from float into eight-bit equivalents. See '
                     'documentation here: '
                     'https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms')
flags.DEFINE_boolean('channels_first', False,
                     'Whether channels are the first dimension of a tensor. '
                     'The default is TensorFlow behaviour where channels are '
                     'the last dimension.')
flags.DEFINE_boolean('output_meta_ckpt', False,
                     'If set to True, exports the model as .meta, .index, and '
                     '.data files, with a checkpoint file. These can be later '
                     'loaded in TensorFlow to continue training.')

flags.mark_flag_as_required('input_model')
flags.mark_flag_as_required('output_model')


def load_model(input_model_path, input_json_path):
    if not Path(input_model_path).exists():
        raise FileNotFoundError(
            'Model file `{}` does not exist.'.format(input_model_path))
    try:
        # 下面一行已经修改，在改回普通的Keras加载模型时，需要去掉custom_objects
        model = keras.models.load_model(input_model_path, custom_objects=custom_objects)
        return model
    except FileNotFoundError as err:
        logging.error('Input mode file (%s) does not exist.', FLAGS.input_model)
        raise err
    except ValueError as wrong_file_err:
        if input_json_path:
            if not Path(input_json_path).exists():
                raise FileNotFoundError(
                    'Model description json file `{}` does not exist.'.format(
                        input_json_path))
            try:
                model = model_from_json(open(str(input_json_path)).read())
                model.load_weights(input_model_path)
                return model
            except Exception as err:
                logging.error("Couldn't load model from json.")
                raise err
        else:
            logging.error(
                'Input file specified only holds the weights, and not '
                'the model definition. Save the model using '
                'model.save(filename.h5) which will contain the network '
                'architecture as well as its weights. If the model is '
                'saved using model.save_weights(filename), the flag '
                'input_model_json should also be set to the '
                'architecture which is exported separately in a '
                'json format. Check the keras documentation for more details '
                '(https://keras.io/getting-started/faq/)')
            raise wrong_file_err


def main(args):
    logging.info("begin====================================================")
    # If output_model path is relative and in cwd, make it absolute from root
    output_model = FLAGS.output_model
    if str(Path(output_model).parent) == '.':
        output_model = str((Path.cwd() / output_model))

    output_fld = Path(output_model).parent
    output_model_name = Path(output_model).name
    output_model_stem = Path(output_model).stem
    output_model_pbtxt_name = output_model_stem + '.pbtxt'

    # Create output directory if it does not exist
    # print (Path(output_model).parent)
    if not os.path.exists(str(Path(output_model).parent)):
        Path(output_model).parent.mkdir(parents=True)

    if FLAGS.channels_first:
        K.set_image_data_format('channels_first')
    else:
        K.set_image_data_format('channels_last')

    model = load_model(FLAGS.input_model, FLAGS.input_model_json)

    input_node_names = [node.op.name for node in model.inputs]
    logging.info('Input nodes names are: %s', str(input_node_names))

    # TODO(amirabdi): Support networks with multiple inputs
    orig_output_node_names = [node.op.name for node in model.outputs]
    if FLAGS.output_nodes_prefix:  # 给模型节点编号
        num_output = len(orig_output_node_names)
        pred = [None] * num_output
        converted_output_node_names = [None] * num_output

        # Create dummy tf nodes to rename output
        for i in range(num_output):
            converted_output_node_names[i] = '{}{}'.format(
                FLAGS.output_nodes_prefix, i)
            pred[i] = tf.identity(model.outputs[i],
                                  name=converted_output_node_names[i])
    else:
        converted_output_node_names = orig_output_node_names
    logging.info('Converted output node names are: %s',
                 str(converted_output_node_names))

    sess = K.get_session()
    if FLAGS.output_meta_ckpt:  # 让转化的模型可以继续被训练
        saver = tf.train.Saver()
        saver.save(sess, str(output_fld / output_model_stem))

    if FLAGS.save_graph_def:  # 以ascii形式存储模型
        tf.train.write_graph(sess.graph.as_graph_def(), str(output_fld),
                             output_model_pbtxt_name, as_text=True)
        logging.info('Saved the graph definition in ascii format at %s',
                     str(Path(output_fld) / output_model_pbtxt_name))

    if FLAGS.quantize:  # 将权重从float转为八位比特
        from tensorflow.tools.graph_transforms import TransformGraph
        transforms = ["quantize_weights", "quantize_nodes"]
        transformed_graph_def = TransformGraph(sess.graph.as_graph_def(), [],
                                               converted_output_node_names,
                                               transforms)
        constant_graph = graph_util.convert_variables_to_constants(
            sess,
            transformed_graph_def,
            converted_output_node_names)
    else:  # float形式存储权重
        constant_graph = graph_util.convert_variables_to_constants(
            sess,
            sess.graph.as_graph_def(),
            converted_output_node_names)

    graph_io.write_graph(constant_graph, str(output_fld), output_model_name,
                         as_text=False)
    logging.info('Saved the freezed graph at %s',
                 str(Path(output_fld) / output_model_name))


if __name__ == "__main__":
    app.run(main)