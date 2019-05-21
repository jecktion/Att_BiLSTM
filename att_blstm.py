# coding=utf-8
# @author: wgs
# blog: https://blog.csdn.net/Kaiyuan_sjtu

import tensorflow as tf
FLAGS = tf.flags.FLAGS

def attention(inputs):
    # Trainable parameters
    hidden_size = inputs.shape[2].value
    u_omega = tf.get_variable("u_omega", [hidden_size], initializer=tf.keras.initializers.glorot_normal())

    with tf.name_scope('v'):
        v = tf.tanh(inputs)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    # Final output with tanh
    output = tf.tanh(output)

    return output, alphas


class ATT_BLSTM:
    def __init__(self, sequence_len, num_classes, vocab_dim, embed_dim, hidden_size, l2_reg=0.0):

        # set placeholder
        self.input_text = tf.placeholder(tf.int32, shape=[None, sequence_len], name='input_text')
        self.input_y = tf.placeholder(tf.int32, shape=[None, num_classes], name='input_y')
        self.embed_dropout_keep_prob = tf.placeholder(tf.float32, name='embed_dropout_keep_prob')
        self.rnn_dropout_keep_prob = tf.placeholder(tf.float32, name='rnn_dropout_keep_prob')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        initializer = tf.keras.initializers.glorot_normal

        # word embedding
        with tf.variable_scope('word_embedding'):
            self.W_text = tf.Variable(tf.random_uniform([vocab_dim, embed_dim], -0.25, 0.25), name='W_text')
            self.embedded_chars = tf.nn.embedding_lookup(self.W_text, self.input_text)

        # dropout
        with tf.variable_scope('embedding_dropout'):
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.embed_dropout_keep_prob)

        # bilstm
        with tf.variable_scope('bi-lstm'):
            _fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, initializer=initializer())
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(_fw_cell, self.rnn_dropout_keep_prob)
            _bw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, initializer=initializer())
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(_bw_cell, self.rnn_dropout_keep_prob)
            self.rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, self.embedded_chars,
                                                                  self._length(self.input_text),
                                                                  dtype=tf.float32)
            self.rnn_outputs = tf.add(self.rnn_outputs[0], self.rnn_outputs[1])

        # attention
        with tf.variable_scope('attention'):
            self.attention, self.alpha = attention(self.rnn_outputs)

        # dropout
        with tf.variable_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.attention, self.dropout_keep_prob)

        # FC layer
        with tf.variable_scope('output'):
            self.logits = tf.layers.dense(self.h_drop, num_classes, kernel_initializer=initializer())
            self.prediction = tf.argmax(self.logits, 1, name='predictions')

        # loss
        with tf.variable_scope('loss'):
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            self.loss = tf.reduce_mean(loss) + l2_reg * self.l2

        # accuracy
        with tf.variable_scope('accuracy'):
            correct_prediction = tf.equal(self.prediction, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    @staticmethod
    def _length(seq):
        relevant = tf.sign(tf.abs(seq))
        length = tf.reduce_sum(relevant, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length
