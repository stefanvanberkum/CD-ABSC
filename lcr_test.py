#!/usr/bin/env python
# encoding: utf-8

# Left-Center-Right LSTMs are fixed.

import os
import sys

sys.path.append(os.getcwd())
from sklearn.metrics import precision_score, recall_score, f1_score
from nn_layer import softmax_layer, bi_dynamic_rnn, reduce_mean_with_len
from att_layer import bilinear_attention_layer, dot_produce_attention_layer
from config import *
from utils import load_w2v, batch_index, load_inputs_twitter_keep
import numpy as np

tf.set_random_seed(1)


def lcr_rot(input_fw, input_bw, sen_len_fw, sen_len_bw, target, sen_len_tr, keep_prob1, keep_prob2, l2, _id='all'):
    print('I am lcr_rot_alt from Maria tester.')
    cell = tf.contrib.rnn.LSTMCell
    # left hidden
    input_fw = tf.nn.dropout(input_fw, keep_prob=keep_prob1)
    hiddens_l = bi_dynamic_rnn(cell, input_fw, FLAGS.n_hidden, sen_len_fw, FLAGS.max_sentence_len, 'l' + _id, 'all')
    pool_l = reduce_mean_with_len(hiddens_l, sen_len_fw)

    # right hidden
    input_bw = tf.nn.dropout(input_bw, keep_prob=keep_prob1)
    hiddens_r = bi_dynamic_rnn(cell, input_bw, FLAGS.n_hidden, sen_len_bw, FLAGS.max_sentence_len, 'r' + _id, 'all')
    pool_r = reduce_mean_with_len(hiddens_r, sen_len_bw)

    # target hidden
    target = tf.nn.dropout(target, keep_prob=keep_prob1)
    hiddens_t = bi_dynamic_rnn(cell, target, FLAGS.n_hidden, sen_len_tr, FLAGS.max_sentence_len, 't' + _id, 'all')
    pool_t = reduce_mean_with_len(hiddens_t, sen_len_tr)

    # attention left
    att_l = bilinear_attention_layer(hiddens_l, pool_t, sen_len_fw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'tl')
    outputs_t_l_init = tf.matmul(att_l, hiddens_l)
    outputs_t_l = tf.squeeze(outputs_t_l_init)
    # attention right
    att_r = bilinear_attention_layer(hiddens_r, pool_t, sen_len_bw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base, 'tr')
    outputs_t_r_init = tf.matmul(att_r, hiddens_r)
    outputs_t_r = tf.squeeze(outputs_t_r_init)

    # attention target left
    att_t_l = bilinear_attention_layer(hiddens_t, outputs_t_l, sen_len_tr, 2 * FLAGS.n_hidden, l2, FLAGS.random_base,
                                       'l')
    outputs_l_init = tf.matmul(att_t_l, hiddens_t)
    outputs_l = tf.squeeze(outputs_l_init)
    # attention target right
    att_t_r = bilinear_attention_layer(hiddens_t, outputs_t_r, sen_len_tr, 2 * FLAGS.n_hidden, l2, FLAGS.random_base,
                                       'r')
    outputs_r_init = tf.matmul(att_t_r, hiddens_t)
    outputs_r = tf.squeeze(outputs_r_init)

    outputs_init_context = tf.concat([outputs_t_l_init, outputs_t_r_init], 1)
    outputs_init_target = tf.concat([outputs_l_init, outputs_r_init], 1)
    att_outputs_context = dot_produce_attention_layer(outputs_init_context, None, 2 * FLAGS.n_hidden, l2,
                                                      FLAGS.random_base, 'fin1')
    att_outputs_target = dot_produce_attention_layer(outputs_init_target, None, 2 * FLAGS.n_hidden, l2,
                                                     FLAGS.random_base, 'fin2')
    outputs_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:, :, 0], 2), outputs_l_init))
    outputs_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:, :, 1], 2), outputs_r_init))
    outputs_t_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:, :, 0], 2), outputs_t_l_init))
    outputs_t_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:, :, 1], 2), outputs_t_r_init))

    for i in range(2):
        # attention target
        att_l = bilinear_attention_layer(hiddens_l, outputs_l, sen_len_fw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base,
                                         'tl' + str(i))
        outputs_t_l_init = tf.matmul(att_l, hiddens_l)
        outputs_t_l = tf.squeeze(outputs_t_l_init)

        att_r = bilinear_attention_layer(hiddens_r, outputs_r, sen_len_bw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base,
                                         'tr' + str(i))
        outputs_t_r_init = tf.matmul(att_r, hiddens_r)
        outputs_t_r = tf.squeeze(outputs_t_r_init)

        # attention left
        att_t_l = bilinear_attention_layer(hiddens_t, outputs_t_l, sen_len_tr, 2 * FLAGS.n_hidden, l2,
                                           FLAGS.random_base, 'l' + str(i))
        outputs_l_init = tf.matmul(att_t_l, hiddens_t)
        outputs_l = tf.squeeze(outputs_l_init)

        # attention right
        att_t_r = bilinear_attention_layer(hiddens_t, outputs_t_r, sen_len_tr, 2 * FLAGS.n_hidden, l2,
                                           FLAGS.random_base, 'r' + str(i))
        outputs_r_init = tf.matmul(att_t_r, hiddens_t)
        outputs_r = tf.squeeze(outputs_r_init)

        outputs_init_context = tf.concat([outputs_t_l_init, outputs_t_r_init], 1)
        outputs_init_target = tf.concat([outputs_l_init, outputs_r_init], 1)
        att_outputs_context = dot_produce_attention_layer(outputs_init_context, None, 2 * FLAGS.n_hidden, l2,
                                                          FLAGS.random_base, 'fin1' + str(i))
        att_outputs_target = dot_produce_attention_layer(outputs_init_target, None, 2 * FLAGS.n_hidden, l2,
                                                         FLAGS.random_base, 'fin2' + str(i))
        outputs_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:, :, 0], 2), outputs_l_init))
        outputs_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:, :, 1], 2), outputs_r_init))
        outputs_t_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:, :, 0], 2), outputs_t_l_init))
        outputs_t_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:, :, 1], 2), outputs_t_r_init))

    outputs_fin = tf.concat([outputs_l, outputs_r, outputs_t_l, outputs_t_r], 1)
    prob = softmax_layer(outputs_fin, 8 * FLAGS.n_hidden, FLAGS.random_base, keep_prob2, l2, FLAGS.n_class)
    return prob, att_l, att_r, att_t_l, att_t_r


def main(test_path, accuracyOnt, test_size, remaining_size, learning_rate=0.09, keep_prob=0.3,
         momentum=0.85, l2=0.00001):
    print_config()
    with tf.device('/gpu:1'):
        # Separate train and test embeddings.
        # word_id_mapping, w2v = load_w2v(FLAGS.embedding_path, FLAGS.embedding_dim)
        # word_embedding = tf.constant(w2v, name='word_embedding')
        if FLAGS.train_embedding == FLAGS.test_embedding:
            train_word_id_mapping, train_w2v = load_w2v(FLAGS.train_embedding, FLAGS.embedding_dim)
            train_word_embedding = tf.constant(train_w2v, dtype=np.float32, name='train_word_embedding')
            test_word_id_mapping = train_word_id_mapping
            test_w2v = train_w2v
            test_word_embedding = train_word_embedding
        else:
            train_word_id_mapping, train_w2v = load_w2v(FLAGS.train_embedding, FLAGS.embedding_dim)
            train_word_embedding = tf.constant(train_w2v, dtype=np.float32, name='train_word_embedding')
            test_word_id_mapping, test_w2v = load_w2v(FLAGS.test_embedding, FLAGS.embedding_dim)
            test_word_embedding = tf.constant(test_w2v, dtype=np.float32, name='test_word_embedding')

        keep_prob1 = tf.placeholder(tf.float32)
        keep_prob2 = tf.placeholder(tf.float32)

        with tf.name_scope('inputs'):
            x = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])
            y = tf.placeholder(tf.float32, [None, FLAGS.n_class])
            sen_len = tf.placeholder(tf.int32, None)

            x_bw = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])
            sen_len_bw = tf.placeholder(tf.int32, [None])

            target_words = tf.placeholder(tf.int32, [None, FLAGS.max_target_len])
            tar_len = tf.placeholder(tf.int32, [None])

        # Educated guess that this is for training.
        inputs_fw = tf.nn.embedding_lookup(train_word_embedding, x)
        inputs_bw = tf.nn.embedding_lookup(train_word_embedding, x_bw)
        target = tf.nn.embedding_lookup(train_word_embedding, target_words)

        alpha_fw, alpha_bw = None, None
        prob, alpha_fw, alpha_bw, alpha_t_l, alpha_t_r = lcr_rot(inputs_fw, inputs_bw, sen_len, sen_len_bw, target,
                                                                 tar_len, keep_prob1, keep_prob2, l2, 'all')

        loss = loss_func(y, prob)
        acc_num, acc_prob = acc_func(y, prob)

        # optimizer = train_func(loss, FLAGS.learning_rate, global_step)
        true_y = tf.argmax(y, 1)
        pred_y = tf.argmax(prob, 1)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        import time
        timestamp = str(int(time.time()))
        test_loss = tf.placeholder(tf.float32)
        test_acc = tf.placeholder(tf.float32)
        # train_summary_op, test_summary_op, validate_summary_op, train_summary_writer, test_summary_writer, \
        # validate_summary_writer = summary_func(loss, acc_prob, test_loss, test_acc, _dir, title, sess)

        # save_dir = 'temp_model/' + str(timestamp) + '_' + title + '/'
        # saver = saver_func(save_dir)

        # sess.run(tf.global_variables_initializer())
        # saver.restore(sess, '/-')

        save_dir = "model/restaurant_laptop/"
        saver = saver_func(save_dir)
        saver.restore(sess, save_dir)

        if FLAGS.is_r == '1':
            is_r = True
        else:
            is_r = False

        # Mapping from restaurant training.
        y_onehot_mapping = {'-1': 0, '1': 1, '0': 2}

        if FLAGS.writable == 1:
            with open(FLAGS.results_file, "a") as results:
                results.write("Test data. ")
        te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_y, te_target_word, te_tar_len, _, _, _, _ = load_inputs_twitter_keep(
            test_path,
            y_onehot_mapping,
            test_word_id_mapping,
            FLAGS.max_sentence_len,
            'TC',
            is_r,
            FLAGS.max_target_len,
            pos_neu_neg=True
        )

        def get_batch_data(x_f, sen_len_f, x_b, sen_len_b, yi, target, tl, batch_size, kp1, kp2, is_shuffle=True):
            for index in batch_index(len(yi), batch_size, 1, is_shuffle):
                feed_dict = {
                    x: x_f[index],
                    x_bw: x_b[index],
                    y: yi[index],
                    sen_len: sen_len_f[index],
                    sen_len_bw: sen_len_b[index],
                    target_words: target[index],
                    tar_len: tl[index],
                    keep_prob1: kp1,
                    keep_prob2: kp2,
                }
                yield feed_dict, len(index)

        acc, cost, cnt = 0., 0., 0
        fw, bw, tl, tr, ty, py = [], [], [], [], [], []
        p = []
        for test, num in get_batch_data(te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_y,
                                        te_target_word, te_tar_len, 2000, 1.0, 1.0, False):
            if FLAGS.method == 'TD-ATT' or FLAGS.method == 'IAN':
                _loss, _acc, _fw, _bw, _tl, _tr, _ty, _py, _p = sess.run(
                    [loss, acc_num, alpha_fw, alpha_bw, alpha_t_l, alpha_t_r, true_y, pred_y, prob], feed_dict=test)
                fw += list(_fw)
                bw += list(_bw)
                tl += list(_tl)
                tr += list(_tr)
            else:
                _loss, _acc, _ty, _py, _p, _fw, _bw, _tl, _tr = sess.run(
                    [loss, acc_num, true_y, pred_y, prob, alpha_fw, alpha_bw, alpha_t_l, alpha_t_r], feed_dict=test)
            ty = np.asarray(_ty)
            py = np.asarray(_py)
            p = np.asarray(_p)
            fw = np.asarray(_fw)
            bw = np.asarray(_bw)
            tl = np.asarray(_tl)
            tr = np.asarray(_tr)
            acc += _acc
            cost += _loss * num
            cnt += num
        print('All samples={}, correct prediction={}'.format(cnt, acc))
        acc = acc / cnt
        totalacc = ((acc * remaining_size) + (accuracyOnt * (test_size - remaining_size))) / test_size
        cost = cost / cnt
        print('Test: mini-batch loss={:.6f}, test acc={:.6f}, combined acc={:.6f}'.format(
            cost,
            acc,
            totalacc))

        if FLAGS.writable == 1:
            with open(FLAGS.results_file, "a") as results:
                results.write(
                    "---\nLCR-Rot-Hop++. Test accuracy: {:.6f}, Combined accuracy: {:.6f}\n---\n".format(acc, totalacc))

        P = precision_score(ty, py, average=None)
        R = recall_score(ty, py, average=None)
        F1 = f1_score(ty, py, average=None)
        print('P:', P, 'avg=', sum(P) / FLAGS.n_class)
        print('R:', R, 'avg=', sum(R) / FLAGS.n_class)
        print('F1:', F1, 'avg=', sum(F1) / FLAGS.n_class)

        # print('Learning_rate={}, iter_num={}, batch_size={}, hidden_num={}, l2={}'.format(
        #    FLAGS.learning_rate,
        #    FLAGS.n_iter,
        #    FLAGS.batch_size,
        #    FLAGS.n_hidden,
        #    FLAGS.l2_reg
        # ))

        # Record prediction-realization.
        FLAGS.pos = y_onehot_mapping['1']
        FLAGS.neu = y_onehot_mapping['0']
        FLAGS.neg = y_onehot_mapping['-1']
        pos_count = 0
        neg_count = 0
        neu_count = 0
        pos_correct = 0
        neg_correct = 0
        neu_correct = 0
        for i in range(0, len(ty)):
            if ty[i] == FLAGS.pos:
                # Positive sentiment.
                pos_count += 1
                if py[i] == FLAGS.pos:
                    pos_correct += 1
            elif ty[i] == FLAGS.neu:
                # Neutral sentiment.
                neu_count += 1
                if py[i] == FLAGS.neu:
                    neu_correct += 1
            else:
                # Negative sentiment.
                neg_count += 1
                if py[i] == FLAGS.neg:
                    neg_correct += 1
        if FLAGS.writable == 1:
            with open(FLAGS.results_file, "a") as results:
                results.write("Test results.\n")
                results.write(
                    "Positive. Correct: {}, Incorrect: {}, Total: {}\n".format(pos_correct, pos_count - pos_correct,
                                                                               pos_count))
                results.write(
                    "Neutral. Correct: {}, Incorrect: {}, Total: {}\n".format(neu_correct, neu_count - neu_correct,
                                                                              neu_count))
                results.write(
                    "Negative. Correct: {}, Incorrect: {}, Total: {}\n---\n".format(neg_correct,
                                                                                    neg_count - neg_correct,
                                                                                    neg_count))

        return acc, np.where(np.subtract(py, ty) == 0, 0,
                             1), fw.tolist(), bw.tolist(), tl.tolist(), tr.tolist()


if __name__ == '__main__':
    tf.app.run()
