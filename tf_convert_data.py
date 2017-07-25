import tensorflow as tf

from datasets import LisaTrafficLight_to_tfrecords
from datasets import kitti_to_tfrecords
from datasets import smartphone_traffic_to_tfrecords

def main(_):

    #LisaTrafficLight_to_tfrecords.run('/home/gpu_server2/DataSet/dayTrain/', './tf_records', shuffling = True)
    # kitti_to_tfrecords.run('/home/gpu_server2/DataSet/kitti/data_object_image_2/training/', './tf_records', 'kitti_train', shuffling = True)
    smartphone_traffic_to_tfrecords.run('/media/gpu_server2/Windows/data/','tf_records','mobiletraffic_train', shuffling = True)

if __name__ == '__main__':
    tf.app.run()
