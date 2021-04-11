#!/usr/bin/env python
# encoding: utf-8

# Configuration file.
#
# https://github.com/stefanvanberkum/CD-ABSC
#
# Adapted from Trusca, Wassenberg, Frasincar and Dekker (2020).
# https://github.com/mtrusca/HAABSA_PLUS_PLUS
#
# Truşcǎ M.M., Wassenberg D., Frasincar F., Dekker R. (2020) A Hybrid Approach for Aspect-Based Sentiment Analysis Using
# Deep Contextual Word Embeddings and Hierarchical Attention. In: Bielikova M., Mikkonen T., Pautasso C. (eds) Web
# Engineering. ICWE 2020. Lecture Notes in Computer Science, vol 12128. Springer, Cham.
# https://doi.org/10.1007/978-3-030-50578-3_25

import sys

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# General variables.
tf.app.flags.DEFINE_string("source_domain", "laptop", "source domain (training set)")
tf.app.flags.DEFINE_string("target_domain", "laptop", "target domain (test set)")
tf.app.flags.DEFINE_integer("year", 2014, "year data set")
tf.app.flags.DEFINE_string("embedding_type", "BERT", "type of embedding used (BERT or glove)")
tf.app.flags.DEFINE_integer('embedding_dim', 768, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('batch_size', 20, 'number of example per batch')
tf.app.flags.DEFINE_integer('n_hidden', 300, 'number of hidden unit')
tf.app.flags.DEFINE_float('learning_rate', 0.07, 'learning rate')
tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
tf.app.flags.DEFINE_integer('max_sentence_len', 80, 'max number of tokens per sentence')
tf.app.flags.DEFINE_integer('max_doc_len', 20, 'max number of tokens per sentence')
tf.app.flags.DEFINE_float('l2_reg', 0.00001, 'l2 regularization')
tf.app.flags.DEFINE_float('random_base', 0.01, 'initial random base')
tf.app.flags.DEFINE_integer('display_step', 4, 'number of test display step')
tf.app.flags.DEFINE_integer('n_iter', 200, 'number of train iter')
tf.app.flags.DEFINE_float('keep_prob1', 0.5, 'dropout keep prob')
tf.app.flags.DEFINE_float('keep_prob2', 0.5, 'dropout keep prob')
tf.app.flags.DEFINE_string('t1', 'last', 'type of hidden output')
tf.app.flags.DEFINE_string('t2', 'last', 'type of hidden output')
tf.app.flags.DEFINE_integer('n_layer', 3, 'number of stacked rnn')
tf.app.flags.DEFINE_string('is_r', '1', 'prob')
tf.app.flags.DEFINE_integer('max_target_len', 19, 'max target length')
tf.app.flags.DEFINE_integer('splits', 9, 'number of data splits for test')
tf.app.flags.DEFINE_float('momentum', 0.85, 'momentum rate')
tf.app.flags.DEFINE_integer('pos', 0, 'mapping for positive sentiment')
tf.app.flags.DEFINE_integer('neu', 1, 'mapping for neutral sentiment')
tf.app.flags.DEFINE_integer('neg', 2, 'mapping for negative sentiment')

# Traindata, testdata and single-domain embeddings.
tf.app.flags.DEFINE_string("train_path", "data/programGeneratedData/" + str(
    FLAGS.embedding_dim) + FLAGS.source_domain + "_train_" + str(FLAGS.year) + ".txt", "train data path")
tf.app.flags.DEFINE_string("test_path", "data/programGeneratedData/" + str(
    FLAGS.embedding_dim) + FLAGS.target_domain + "_test_" + str(FLAGS.year) + ".txt", "formatted test data path")
tf.app.flags.DEFINE_string("embedding_path",
                           "data/programGeneratedData/" + str(FLAGS.embedding_dim) + "embedding" + str(
                               FLAGS.year) + ".txt", "pre-trained embedding vectors file path")
tf.app.flags.DEFINE_string("remaining_test_path",
                           "data/programGeneratedData/" + str(FLAGS.embedding_dim) + 'remainingtestdata' + str(
                               FLAGS.year) + ".txt", "formatted remaining test data path after ontology")

# Cross-domain source and target embedding
tf.app.flags.DEFINE_string("train_embedding",
                           "data/programGeneratedData/" + FLAGS.embedding_type + "_" + FLAGS.source_domain + "_" + str(
                               FLAGS.year) + "_" + str(FLAGS.embedding_dim) + ".txt",
                           "source domain pre-trained BERT embeddings")
tf.app.flags.DEFINE_string("test_embedding",
                           "data/programGeneratedData/" + FLAGS.embedding_type + "_" + FLAGS.target_domain + "_" + str(
                               FLAGS.year) + "_" + str(FLAGS.embedding_dim) + ".txt",
                           "source domain pre-trained BERT embeddings")

# SVM train and test data.
# NOTE. Not tested, kept from original code.
tf.app.flags.DEFINE_string("train_svm_path",
                           "data/programGeneratedData/" + str(FLAGS.embedding_dim) + 'trainsvmdata' + str(
                               FLAGS.year) + ".txt", "train data path")
tf.app.flags.DEFINE_string("test_svm_path",
                           "data/programGeneratedData/" + str(FLAGS.embedding_dim) + 'testsvmdata' + str(
                               FLAGS.year) + ".txt", "formatted test data path")
tf.app.flags.DEFINE_string("remaining_svm_test_path",
                           "data/programGeneratedData/" + str(FLAGS.embedding_dim) + 'remainingsvmtestdata' + str(
                               FLAGS.year) + ".txt", "formatted remaining test data path after ontology")

# Hyperparameter tuning train and test data.
tf.app.flags.DEFINE_string("hyper_train_path",
                           "data/programGeneratedData/" + str(FLAGS.embedding_dim) + 'hypertraindata' + str(
                               FLAGS.year) + ".txt", "hyper train data path")
tf.app.flags.DEFINE_string("hyper_eval_path",
                           "data/programGeneratedData/" + str(FLAGS.embedding_dim) + 'hyperevaldata' + str(
                               FLAGS.year) + ".txt", "hyper eval data path")

tf.app.flags.DEFINE_string("hyper_svm_train_path",
                           "data/programGeneratedData/" + str(FLAGS.embedding_dim) + 'hypertrainsvmdata' + str(
                               FLAGS.year) + ".txt", "hyper train svm data path")
tf.app.flags.DEFINE_string("hyper_svm_eval_path",
                           "data/programGeneratedData/" + str(FLAGS.embedding_dim) + 'hyperevalsvmdata' + str(
                               FLAGS.year) + ".txt", "hyper eval svm data path")

# External data sources.
tf.app.flags.DEFINE_string("pretrain_file", "data/externalData/glove.42B." + str(FLAGS.embedding_dim) + "d.txt",
                           "pre-trained glove vectors file path")
tf.app.flags.DEFINE_string("train_data",
                           "data/externalData/" + FLAGS.source_domain + "_train_" + str(FLAGS.year) + ".xml",
                           "train data path")
tf.app.flags.DEFINE_string("test_data",
                           "data/externalData/" + FLAGS.target_domain + "_test_" + str(FLAGS.year) + ".xml",
                           "test data path")

tf.app.flags.DEFINE_string('method', 'AE', 'model type: AE, AT or AEAT')
tf.app.flags.DEFINE_string('prob_file', 'prob', 'prob')

# Test results.
tf.app.flags.DEFINE_string("results_file", "data/programGeneratedData/" + str(
    FLAGS.embedding_dim) + "results_" + FLAGS.source_domain + "_" + FLAGS.target_domain + "_" + str(
    FLAGS.year) + ".txt", "results for test")
tf.app.flags.DEFINE_integer("writable", 0, "one if writable")
tf.app.flags.DEFINE_integer("savable", 0, "one if savable")


def print_config():
    """
    Method obtained from Trusca et al. (2020), no original docstring provided.
    """
    FLAGS(sys.argv)
    print('\nParameters:')
    for k, v in sorted(tf.app.flags.FLAGS.flag_values_dict().items()):
        print('{}={}'.format(k, v))
    print()


def loss_func(y, prob):
    """
    Method obtained from Trusca et al. (2020), no original docstring provided.

    :param y:
    :param prob:
    :return:
    """
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = - tf.reduce_mean(y * tf.log(prob)) + sum(reg_loss)
    return loss


def acc_func(y, prob):
    """
    Method obtained from Trusca et al. (2020), no original docstring provided.

    :param y:
    :param prob:
    :return:
    """
    correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1))
    acc_num = tf.reduce_sum(tf.cast(correct_pred, tf.int32))
    acc_prob = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return acc_num, acc_prob


def summary_func(loss, acc, test_loss, test_acc, _dir, title, sess):
    """
    Method obtained from Trusca et al. (2020), no original docstring provided.
    NOTE. Not used in current adaptation.

    :param loss:
    :param acc:
    :param test_loss:
    :param test_acc:
    :param _dir:
    :param title:
    :param sess:
    :return:
    """
    summary_loss = tf.summary.scalar('loss' + title, loss)
    summary_acc = tf.summary.scalar('acc' + title, acc)
    test_summary_loss = tf.summary.scalar('loss' + title, test_loss)
    test_summary_acc = tf.summary.scalar('acc' + title, test_acc)
    train_summary_op = tf.summary.merge([summary_loss, summary_acc])
    validate_summary_op = tf.summary.merge([summary_loss, summary_acc])
    test_summary_op = tf.summary.merge([test_summary_loss, test_summary_acc])
    train_summary_writer = tf.summary.FileWriter(_dir + '/train', sess.graph)
    test_summary_writer = tf.summary.FileWriter(_dir + '/test')
    validate_summary_writer = tf.summary.FileWriter(_dir + '/validate')
    return train_summary_op, test_summary_op, validate_summary_op, train_summary_writer, test_summary_writer, \
           validate_summary_writer


def saver_func(_dir):
    """
    Method obtained from Trusca et al. (2020), no original docstring provided.

    :param _dir:
    :return:
    """
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=1000)
    import os
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    return saver
