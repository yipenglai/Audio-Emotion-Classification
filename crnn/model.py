from layers import attention, batch_norm_wrapper, leaky_relu
from utils import get_dict
import tensorflow as tf

def acrnn(inputs, 
          num_classes=7,
          is_training=True,
          L1=128,
          L2=256,
          cell_units=128,
          num_linear=768,
          p=10,
          time_step=150,
          F1=64,
          dropout_keep_prob=1):
    """
    Attention-based convolutional recurrent neural network
    Adapted from https://github.com/xuanjihe/speech-emotion-recognition/blob/master/model.py

    Mingyi Chen, Xuanji He, Jing Yang, Han Zhang,
    "3-D Convolutional Recurrent Neural Networks With Attention Model for Speech Emotion Recognition",
    IEEE Signal Processing Letters, vol. 25, no. 10, pp. 1440-1444, 2018.
    """
    # Fetch filter, weights and bias
    filters, weights, bias = get_dict(num_classes, L1, L2, cell_units, num_linear, F1, p)
    # Covolutional layer 1
    conv1 = tf.nn.conv2d(inputs, filters["conv1"], strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.bias_add(conv1, bias["conv1"])
    conv1 = leaky_relu(conv1, 0.01)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='VALID', name='max_pool')
    conv1 = tf.contrib.layers.dropout(conv1, keep_prob=dropout_keep_prob, is_training=is_training)
    # layer1: [batch_size, 150, 10, 128]

    # Convolutional layer 2
    conv2 = tf.nn.conv2d(conv1, filters["conv2"], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.bias_add(conv2, bias["conv2"])
    conv2 = leaky_relu(conv2, 0.01)
    conv2 = tf.contrib.layers.dropout(conv2, keep_prob=dropout_keep_prob, is_training=is_training)
    conv2 = tf.reshape(conv2,[-1,time_step,L2*p])
    conv2 = tf.reshape(conv2, [-1,p*L2])
    # layer2: [None, 2560]

    # Linear layer
    linear1 = tf.matmul(conv2, weights["linear1"]) + bias["linear1"]
    linear1 = batch_norm_wrapper(linear1, is_training)
    linear1 = leaky_relu(linear1, 0.01)
    linear1 = tf.reshape(linear1, [-1, time_step, num_linear])

    # LSTM layer
    # Forward direction cell
    gru_fw_cell1 = tf.contrib.rnn.BasicLSTMCell(cell_units, forget_bias=1.0)
    # Backward direction cell
    gru_bw_cell1 = tf.contrib.rnn.BasicLSTMCell(cell_units, forget_bias=1.0)
    
    # Now we feed `layer_3` into the LSTM BRNN cell and obtain the LSTM BRNN output.
    outputs1, output_states1 = tf.nn.bidirectional_dynamic_rnn(cell_fw=gru_fw_cell1,
                                                            cell_bw=gru_bw_cell1,
                                                            inputs= linear1,
                                                            dtype=tf.float32,
                                                            time_major=False,
                                                            scope='LSTM1')
    # Attention layer
    gru, alphas = attention(outputs1, 1, return_alphas=True)
    # Fully connected layer
    fully1 = tf.matmul(gru, weights["fully1"]) + bias["fully1"]
    fully1 = leaky_relu(fully1, 0.01)
    fully1 = tf.nn.dropout(fully1, dropout_keep_prob)
    Ylogits = tf.matmul(fully1, weights["fully2"]) + bias["fully2"]
    return Ylogits
