import numpy as np
import tensorflow as tf
import pandas as pd


class ESMM(object):
    def __init__(self, cfg):

        # featGrps是配置文件
        # featSize是所有用到的特征

        self.featGrps, self.featSize = cfg['featGrps'], cfg['featSize']
        if self.featGrps[0]['k'] != -1:
            print('Error! The first group(bypass-variables)\'s k is not -1. Invalid')
            exit(-1)

        self.grpNum = len(self.featGrps)   # 需要做embedding的数量
        self.Xarr = [] 
        self.lutArr = [] 
        self.embArr = [] 
        self.lr = cfg['LEARN_RATE']
        self.adam_epsilon = cfg['ADAM_EPSILON']
        self.Y0_C_L2 = cfg['Y0_C_L2']
        self.Y1_C_L2 = cfg['Y1_C_L2']
        self.Y2_C_L2 = cfg['Y2_C_L2']
        self.MLP_HIDDEN_CONTS = cfg['MLP_HIDDEN_CONTS']

    def build_graph(self):
        print('***************************')
        print('    Build graph started')
        print('***************************')

      # Input placeholders
      # 注意self.x,self.y0...设置了shape，所以他们可以看作是dataframe，而不是pd.Series(因为series的shape是(xx,))
        self.X = tf.placeholder('float32', [None, self.featSize], name='input_X')  # 所有要计算特征
        self.y0 = tf.placeholder('int64', [None, 1], name='input_y0')
        self.y1 = tf.placeholder('int64', [None, 1], name='input_y1')
        self.y2 = tf.placeholder('int64', [None, 1], name='input_y2')

      # X Slices by group
        # featGrps[0]: feas no need to embedding
        print('[input_X -> slices (Xarr) by group]')
        count = self.featGrps[0]['featureNum']
        self.Xby = self.X[:,0:count]
        print('group-%d: start=%d, size=%d   *** X-bypass' %(0, 0, count))


        for gid in range(1, self.grpNum): 
            start = count
            size = self.featGrps[gid]['featureNum']
            # self.Xarr 需要做embedding的特征原始值
            self.Xarr.append(tf.cast(self.X[:,start:start+size], tf.int64))
            count += size
            print('group-%d: start=%d, size=%d' %(gid, start, size))

      # Group Embedding
        print('[LookUpTable] - %d groups' %(self.grpNum-1)) 
        #################
        # 开始做embedding操作
        #################
        for gid in range(1, self.grpNum):
            dim = self.featGrps[gid]['dim']
            k = self.featGrps[gid]['k']
            # 生成一个dim*k, mean=0, stddev=0.01的张量
            self.lutArr.append(tf.Variable(tf.random.normal([dim, k], 0.0, 0.01), name="LUT_%d" %gid)) 
            print('LUT-%d: %d x %d' %(gid, dim, k))

        print('[After embeding] - %d groups' %(self.grpNum-1))
        for i in range(len(self.lutArr)):
            # 特征的每个取值对应的初始embedding 就是emb
            emb = tf.nn.embedding_lookup(self.lutArr[i], self.Xarr[i])

            # mode-0
            size = self.featGrps[i+1]['featureNum'] * self.featGrps[i+1]['k']
            #self.embArr.append(tf.reshape(emb, shape=[-1, size]))  # 对emb进行reshape
            #print('emb-%d: reshape_to_size=%d' %(i+1, size))
            
            # mode-1
            self.embArr.append(tf.reduce_mean(emb, axis=1))   # mode-1 是计算mean
            # mode-2
            #self.embArr.append(tf.reduce_sum(emb, axis=1))

      # Concat all
        self.embArr.append(self.Xby)
        self.mlpIn = tf.concat(self.embArr, axis=-1)  #axis=-1表示按照最高维度进行拼接

        if len(self.MLP_HIDDEN_CONTS) == 0:
          # Y0-MLP  第一塔的网络
            y0_logit = tf.layers.dense(self.mlpIn, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=3), \
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(self.Y0_C_L2, 'y0_logit_l2_term'))
          # Y1-MLP  第二塔的网络
            y1_logit = tf.layers.dense(self.mlpIn, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=6), \
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(self.Y1_C_L2, 'y1_logit_l2_term'))
          # Y2-MLP  第三塔的网络
            y2_logit = tf.layers.dense(self.mlpIn, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=9), \
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(self.Y2_C_L2, 'y2_logit_l2_term'))

        elif len(self.MLP_HIDDEN_CONTS) == 1:
          # Y0-MLP
            y0_layer0 = tf.layers.dense(self.mlpIn, self.MLP_HIDDEN_CONTS[0], activation=tf.nn.sigmoid, \
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=1))
            y0_logit = tf.layers.dense(y0_layer0, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=3), \
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(self.Y0_C_L2, 'y0_logit_l2_term'))
          # Y1-MLP
            y1_layer0 = tf.layers.dense(self.mlpIn, self.MLP_HIDDEN_CONTS[0], activation=tf.nn.sigmoid, \
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=4))
            y1_logit = tf.layers.dense(y1_layer0, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=6), \
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(self.Y1_C_L2, 'y1_logit_l2_term'))
          # Y2-MLP
            y2_layer0 = tf.layers.dense(self.mlpIn, self.MLP_HIDDEN_CONTS[0], activation=tf.nn.sigmoid, \
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=7))
            y2_logit = tf.layers.dense(y2_layer0, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=9), \
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(self.Y2_C_L2, 'y2_logit_l2_term'))

        elif len(self.MLP_HIDDEN_CONTS) == 2:
          # Y0-MLP
            y0_layer0 = tf.layers.dense(self.mlpIn, self.MLP_HIDDEN_CONTS[0], activation=tf.nn.sigmoid, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=1))
            y0_layer1 = tf.layers.dense(y0_layer0, self.MLP_HIDDEN_CONTS[1], activation=tf.nn.sigmoid, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=2))
            y0_logit = tf.layers.dense(y0_layer1, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=3), \
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(self.Y0_C_L2, 'y0_l2_term'))
          # Y1-MLP
            y1_layer0 = tf.layers.dense(self.mlpIn, self.MLP_HIDDEN_CONTS[0], activation=tf.nn.sigmoid, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=4))
            y1_layer1 = tf.layers.dense(y1_layer0, self.MLP_HIDDEN_CONTS[1], activation=tf.nn.sigmoid, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=5))
            y1_logit = tf.layers.dense(y1_layer1, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=6), \
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(self.Y1_C_L2, 'y1_l2_term'))
          # Y2-MLP
            y2_layer0 = tf.layers.dense(self.mlpIn, self.MLP_HIDDEN_CONTS[0], activation=tf.nn.sigmoid, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=7))
            y2_layer1 = tf.layers.dense(y2_layer0, self.MLP_HIDDEN_CONTS[1], activation=tf.nn.sigmoid, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=8))
            y2_logit = tf.layers.dense(y2_layer1, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=9), \
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(self.Y2_C_L2, 'y2_l2_term'))
        else:
            raise ValueError(f'Unknown value')
        
      # Sigmoid  # 整个esmm的结构见文件
        self.y0_pred = tf.nn.sigmoid(y0_logit)  # 做sigmoid操作（很像逻辑回归）
        self.y1_pred = tf.nn.sigmoid(y1_logit) * self.y0_pred
        self.y2_pred = tf.multiply(tf.nn.sigmoid(y2_logit), self.y1_pred, 'y_predict_prob')

      # Loss function
        y0_f = tf.cast(self.y0, tf.float32)
        y1_f = tf.cast(self.y1, tf.float32)
        y2_f = tf.cast(self.y2, tf.float32)
        
        
        # get_collection：根据key或者命名区间范围获取集合的函数 
        # tf.GraphKeys.REGULARIZATION_LOSSES: 正则化产生的损失都放在这里（最终会把正则化损失加到总的损失中）
        l2_set = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        print("*************** [L2 regularizer terms] ***************")
        for k in l2_set:
            print(k.name)
        print("******************************************************")
        
        # loss是这三个loss的总和
        self.loss = tf.losses.log_loss(y0_f, self.y0_pred) + \
              1.0 * tf.losses.log_loss(y1_f, self.y1_pred) + \
              1.0 * tf.losses.log_loss(y2_f, self.y2_pred) + \
              tf.add_n(l2_set)

        tf.summary.scalar('loss', self.loss) # tensorboard use this

      # Accuracy
        self.correct_prediction = tf.equal(tf.cast(tf.greater(self.y2_pred, 0.5), tf.int64), self.y2)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy) # tensorboard use this
        
      # Optimizer 这个就是核心，基本是这样的
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=self.adam_epsilon).minimize(self.loss)
