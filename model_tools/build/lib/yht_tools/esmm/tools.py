import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import re
import datetime as dt
import importlib
import logging
import numpy as np
import os
import pandas as pd
import sys
from glob import glob
import yaml
from os.path import join, split, splitext, basename, exists, abspath, dirname
from sklearn.metrics import roc_curve, auc


def BatchIter(df, cols, target_cols, batch_size=1024):
    """
    batch_size df
    """
    nchunk = round(df.shape[0] / batch_size + 0.5)
    for label_chunk in np.array_split(df, nchunk):
        yield label_chunk[cols].reset_index(drop=True), label_chunk[target_cols].reset_index(drop=True)


def verification_3t(sess, model, df, cols, target_cols, batch_size=10000):
    """
    to verification df performance
    """
    v_loss = 0.0
    v_accu = 0.0
    sCount = 0.0
    y_ks = []
    pred_ks = []

    for batch_x, batch_y in BatchIter(df, cols, target_cols, batch_size):
        X = np.array(batch_x)
        y0 = np.array(batch_y[[target_cols[0]]])
        y1 = np.array(batch_y[[target_cols[1]]])
        y2 = np.array(batch_y[[target_cols[2]]])

        feed_dict = {model.X: X, model.y0: y0, model.y1: y1, model.y2: y2}
        # model.y2_pred, model.loss, model.accuracy 都在esmm的class里面定义过(且都是求平均的, 因此下文需乘以样本数)
        preds, loss, accu = sess.run([model.y2_pred, model.loss, model.accuracy], feed_dict=feed_dict)

        bsize = len(X)
        v_loss += loss * bsize
        v_accu += accu * bsize
        sCount += bsize

        for j in range(bsize):
            y_ks.append(y2[j])
            pred_ks.append(preds[j])

    # calculat the ks
    fpr, tpr, thresholds = roc_curve(y_ks, pred_ks)
    ks = max(tpr - fpr)
    roc_auc = round(auc(fpr, tpr), 4)

    return v_loss / sCount, v_accu / sCount, roc_auc, ks


def train_model_3t(sess, model, method, saver, x_train, validation, CHKPOINTS_PATH,
                   cols, target_cols, batch_size=1024, epochs=300, print_every=1,
                   NO_BETTER_STOP=30):
    """
    如分成10个epochs, 每个epochs里面都是全量的训练数据训练(求平均值)
    每个epochs里面, 又把训练集进行切分, 如每次只放入1024个样本进行训练(并查看模型在valid上的表现)
    """

    # Merge all the summaries and write them out to train_logs
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train_logs', sess.graph)

    # main loop - train
    print('************************************************')
    print('   [Model training start]')
    print('************************************************')

    noBetters = 0
    index_ = 0
    champion = 0
    # epochs
    for e in range(epochs):
        better = False
        num_samples = 0
        losses = 0.0

        for batch_x, batch_y in BatchIter(x_train, cols, target_cols, batch_size):
            batch_x = np.array(batch_x)
            batch_y0 = np.array(batch_y[[target_cols[0]]])
            batch_y1 = np.array(batch_y[[target_cols[1]]])
            batch_y2 = np.array(batch_y[[target_cols[2]]])

            feed_dict = {model.X: batch_x, model.y0: batch_y0, model.y1: batch_y1, model.y2: batch_y2}
            loss, accuracy, summary, _ = sess.run([model.loss, model.accuracy, merged, model.optimizer],
                                                  feed_dict=feed_dict)
            # model.optimizer就是不断优化，所以训练的时候需要加这个，最后输出就是优化好的模型

            curr_batch_size = len(batch_x)
            losses += loss * curr_batch_size
            num_samples += curr_batch_size
            train_writer.add_summary(summary, index_)  # Record summaries

            # 查看在valid上的表现(print_every若为1, 则每训练一次都查看在valid上的效果)
            if index_ % print_every == 0:
                v_loss, v_accu, v_auc, v_ks = verification_3t(sess, model, validation, cols, target_cols, batch_size)
                if method == 'accu':
                    curr_idx = v_accu
                elif method == 'auc':
                    curr_idx = v_auc
                elif method == 'ks':
                    curr_idx = v_ks
                else:
                    raise ValueError('please input wright method')

                tail = ''
                if curr_idx > champion:
                    champion = curr_idx  # 如果结果优于champion, 则用champion替代curr_idx
                    saver.save(sess, CHKPOINTS_PATH + 'model', index_)  # only save the best
                    tail = '*'
                    better = True
                print("Iter %d: train_loss=%3.4f, train_accu=%2.3f%%, verf_loss=%3.4f, \
                verf_accu=%2.3f%% verf_auc=%1.5f verf_ks=%1.5f (best_%s=%1.5f) %s" \
                      % (index_, loss, accuracy * 100, v_loss, v_accu * 100, v_auc, v_ks, method, champion, tail))
            index_ += 1

        total_loss = losses / num_samples  # total loss 一个epoch测试完之后计算平均loss
        print('Epoch %d finished! Overall loss = %3.4f' % (e + 1, total_loss))

        # check whether the training is over
        if better:
            noBetters = 0
        else:  # better=False 一轮epoch运行下来,相较于之前的没有提升
            noBetters += 1
            if noBetters >= NO_BETTER_STOP:
                print('No better verification(%s) found after %d epoches. Training stopped.' % (method, NO_BETTER_STOP))
                break

    return champion


def load_tf_model(sess, saver, CHKPOINTS_PATH):
    ckpt = tf.train.get_checkpoint_state(CHKPOINTS_PATH)

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('Error to load checkpoint from path: "%s"' % CHKPOINTS_PATH)
        exit(-1)


def get_df_pred_3t(sess, model, df, cols, target_cols):
    # load test data
    testX, testY = np.array(df[cols]), np.array(df[target_cols])
    y0 = testY[:, 0:1]
    y1 = testY[:, 1:2]
    y2 = testY[:, 2:3]

    feed_dict = {model.X: testX, model.y0: y0, model.y1: y1, model.y2: y2}
    y0_preds, y1_preds, preds = \
        sess.run([model.y0_pred, model.y1_pred, model.y2_pred], \
                 feed_dict=feed_dict)

    return preds


def read_file(file):
    if file.endswith('.pkl'):
        return pd.read_pickle(file)
    
    elif file.endswith('.csv'):
        return pd.read_csv(file)
    
    else:
        raise ValueError('only support pkl & csv')


def split_df(df, split_size):
    length = df.shape[0]
    split_size = [int(c*length) for c in [sum(split_size[:i]) for i in range(len(split_size))]+[1]]

    out = []
    for i in range(len(split_size)-1):
        tmp = df[split_size[i]:split_size[i+1]]
        out.append(tmp)

    return out