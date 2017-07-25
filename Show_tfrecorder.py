import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datasets import dataset_factory
from notebooks import visualization

slim = tf.contrib.slim

tf.app.flags.DEFINE_string('dataset_name', 'smartphone_traffic', 'The name of the dataset to load.')
tf.app.flags.DEFINE_string('dataset_split_name', 'train', 'The name of train/test split.')
tf.app.flags.DEFINE_string('dataset_dir', 'tf_records', 'The directory where the dataset files are stored.')

FLAGS = tf.app.flags.FLAGS

def _resize_image(image, height, width):
  image = tf.expand_dims(image, 0)
  image = tf.image.resize_bilinear(image, [height, width])
  return tf.squeeze(image, [0])

def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    with tf.Graph().as_default():

        dataset = dataset_factory.get_dataset(FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

        # Creates a TF-Slim DataProvider which reads the dataset in the background
        # during both training and testing.
        provider = slim.dataset_data_provider.DatasetDataProvider(dataset, num_readers = 3,
                                                                  common_queue_capacity = 20, common_queue_min=10)
        [image, shape, glabel, gbboxes] = provider.get(['image', 'shape', 'object/label', 'object/bbox'])


        with tf.Session() as sess:

            with slim.queues.QueueRunners(sess):
                i = 0
                while (i < 10):
                    shape_show = sess.run([shape])
                    print (shape_show[0][0], shape_show[0][1])
                    image_show = _resize_image(image, shape_show[0][0], shape_show[0][1])
                    #tf.reverse(image, axis = [-1])
                    image_show, label, boxes = sess.run([image_show, glabel, gbboxes])
                    print (label, boxes)
                    visualization.plt_bboxes_tfrecorder(image_show, label, boxes)
                    i += 1



if __name__ == '__main__':
    tf.app.run()
