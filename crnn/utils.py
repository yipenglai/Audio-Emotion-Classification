import tensorflow as tf
import numpy as np

def dense_to_one_hot(labels_dense, num_classes):
    """
    Convert class labels from scalars to one-hot vectors.
    """
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

# Store filters, weights and bias
def get_dict(num_classes=7, L1=128, L2=256,
        cell_units=128, num_linear=768, F1=64, p=10):
    """
    Return filters, weights and bias for nn layers
    """
    with tf.variable_scope("get_dict", reuse=tf.AUTO_REUSE):
        filters = {
            # Conv layer 1
            "conv1": tf.get_variable('layer1_filter', shape=[5, 3, 3, L1], dtype=tf.float32, 
                                      initializer=tf.truncated_normal_initializer(stddev=0.1)),
            # Conv layer 2
            "conv2": tf.get_variable('layer2_filter', shape=[5, 3, L1, L2], dtype=tf.float32, 
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        }

        weights = {
            # Linear layer 1
            "linear1": tf.get_variable('linear1_weight', shape=[p*L2,num_linear], dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1)),
            # Fully connected layer 1
            "fully1": tf.get_variable('fully1_weight', shape=[2*cell_units,F1], dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1)),
            # Fully connected layer 2
            "fully2": tf.get_variable('fully2_weight', shape=[F1,num_classes], dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        }

        bias = {
            # Conv layer 1
            "conv1": tf.get_variable('layer1_bias', shape=[L1], dtype=tf.float32,
                                              initializer=tf.constant_initializer(0.1)),
            # Conv layer 2
            "conv2": tf.get_variable('layer2_bias', shape=[L2], dtype=tf.float32,
                                              initializer=tf.constant_initializer(0.1)),
            # Linear layer 1
            "linear1": tf.get_variable('linear1_bias', shape=[num_linear], dtype=tf.float32,
                                              initializer=tf.constant_initializer(0.1)),
            # Fully connected layer 1
            "fully1": tf.get_variable('fully1_bias', shape=[F1], dtype=tf.float32,
                                              initializer=tf.constant_initializer(0.1)),
            # Fully connecte layer 2
            "fully2": tf.get_variable('fully2_bias', shape=[num_classes], dtype=tf.float32,
                                              initializer=tf.constant_initializer(0.1))
        }
        return (filters, weights, bias)
