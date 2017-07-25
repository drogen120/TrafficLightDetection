import os
import tensorflow as tf
import tensorflow.contrib.slim as slim

from nets import mobilenet_ssd_traffic
from nets import nets_factory
from nets import mobilenet_pretrained

from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

data_format = 'NHWC'
#checkpoint_path = tf.train.latest_checkpoint('./train/')
ckpt = tf.train.get_checkpoint_state(os.path.dirname('./logs/checkpoint'))

with tf.Graph().as_default() as graph:
    input_tensor = tf.placeholder(tf.float32, shape=(None, 448, 448, 3), name='input_image')
    with tf.Session() as sess:
        ssd_net = mobilenet_pretrained.Mobilenet_SSD_Traffic()
        with slim.arg_scope(ssd_net.arg_scope(data_format=data_format, is_training=False)):
            predictions, localisations, _, _ = ssd_net.net(input_tensor, is_training=False)

    # saver = tf.train.Saver()
    # saver.restore(sess, checkpoint_path)
    saver = tf.train.Saver()
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    output_node_names = 'MobilenetV1/Box/softmax_3/Reshape_1,MobilenetV1/Box/softmax_2/Reshape_1,MobilenetV1/Box/softmax_1/Reshape_1,MobilenetV1/Box/softmax/Reshape_1'
    input_graph_def = graph.as_graph_def()
    output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def, output_node_names.split(","))
    with open('./output_graph_nodes.txt', 'w') as f:
        for node in output_graph_def.node:
            f.write(node.name + '\n')

    output_graph = './mobilenet_traffic_both.pb'
    with gfile.FastGFile(output_graph, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
