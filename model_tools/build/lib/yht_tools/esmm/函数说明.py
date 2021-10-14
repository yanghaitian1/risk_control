1. yeild

https://www.runoob.com/w3cnote/python-yield-used-analysis.html

一个带有 yield 的函数就是一个 generator，它和普通函数不同，生成一个 generator 看起来像函数调用，
但不会执行任何函数代码，直到对其调用 next()（在 for 循环中会自动调用 next()）才开始执行。
虽然执行流程仍按函数的流程执行，但每执行到一个 yield 语句就会中断，并返回一个迭代值，
下次执行时从 yield 的下一个语句继续执行。看起来就好像一个函数在正常执行的过程中被 yield 中断了数次，
每次中断都会通过 yield 返回当前的迭代值。yield 的好处是显而易见的，把一个函数改写为一个 generator 就获得了迭代能力，
比起用类的实例保存状态来计算下一个 next() 的值，不仅代码简洁，而且执行流程异常清晰。


def fab(max): 
    n, a, b = 0, 0, 1 
    while n < max: 
        yield b
        a, b = b, a + b 
        n = n + 1

fab(3)
for n in fab(3): 
    print(n)

    ---><generator object fab at 0x0000027B9D539748>
    --->1
    --->1
    --->2


2. tf.cast()函数的作用是执行tensorflow中张量数据类型转换(如embedding要先转化成int)

a = tf.cast([1.1,2.5,4], tf.int64)
b = tf.cast(pd.DataFrame([1.1,2.5,4]), tf.int64)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(a),a.shape)
    print('='*10)
    print(sess.run(b),b.shape)
    
    ---> [1 2 4] (3,)
    ---> ============
    ---> [[1]
    --->  [2]
    --->  [4]] (3, 1)


3. tf.Variable(initializer, name) 自定义变量
   参数initializer是初始化参数，name是可自定义的变量名称

    v2=tf.Variable(tf.constant(2), name='v2')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(v2))
    ---> return 2



4. tf.random_normal()函数用于从“服从指定正态分布的序列”中随机取出指定个数的值
tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)

v1=tf.Variable(tf.random_normal(shape=[4,3],mean=0,stddev=1),name='v1')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(v1))
    --->[[ 0.6783711   1.1689452   0.36633876]
    --->[-0.07531247 -0.7322053   0.096654  ]
    --->[ 1.4222699  -0.3466825  -0.69348747]
    --->[-0.01756123 -0.30934024 -0.7680681 ]]


c = tf.Variable(tf.random.normal([3, 1], 0.0, 0.01), name='v1')
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(c))
    --->[[5.3150407e-03]
    ---> [6.1461920e-05]
    ---> [1.8306764e-02]]



5. embedding_lookup

c = np.random.random([5,1])
b = tf.nn.embedding_lookup(c, [0,1,3,4,4,4,2])
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(c,c.shape)
    print('='*10)
    print(sess.run(b),b.shape)
    
    ---> [[0.5079389 ]
    --->  [0.26619519]
    --->  [0.08067123]
    --->  [0.23980668]
    --->  [0.3928173 ]] (5, 1)
    ---> ==========
    ---> [[0.5079389 ]
    --->  [0.26619519]
    --->  [0.23980668]
    --->  [0.3928173 ]
    --->  [0.3928173 ]
    --->  [0.3928173 ]
    --->  [0.08067123]] (7, 1)



c = np.random.random([5,1])
b = tf.nn.embedding_lookup(c, pd.DataFrame([0,1,3,4,4,4,2]))
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(c,c.shape)
    print('='*10)
    print(sess.run(b),b.shape)
    ---> [[0.97686573]
    --->  [0.45437764]
    --->  [0.64116391]
    --->  [0.53445146]
    --->  [0.82573572]] (5, 1)
    ---> ==========
    ---> [[[0.97686573]]
    ---> 
    --->  [[0.45437764]]
    ---> 
    --->  [[0.53445146]]
    ---> 
    --->  [[0.82573572]]
    ---> 
    --->  [[0.82573572]]
    ---> 
    --->  [[0.82573572]]
    ---> 
    --->  [[0.64116391]]] (7, 1, 1)



6. reshape

c = tf.Variable(tf.random.normal([3, 1], 0.0, 0.01), name='v1')
b = tf.nn.embedding_lookup(c, [0,1,2,1,1,0])
d = tf.reshape(b, shape=[-1, 1])
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(b),b.shape)
    print('='*10)
    print(sess.run(d),d.shape)
    --->[[-0.00651558]
    ---> [ 0.00162287]
    ---> [-0.01091311]
    ---> [ 0.00162287]
    ---> [ 0.00162287]
    ---> [-0.00651558]] (6, 1)
    --->==========
    --->[[-0.00651558]
    ---> [ 0.00162287]
    ---> [-0.01091311]
    ---> [ 0.00162287]
    ---> [ 0.00162287]
    ---> [-0.00651558]] (6, 1)


c = tf.Variable(tf.random.normal([3, 1], 0.0, 0.01), name='v1')
b = tf.nn.embedding_lookup(c, pd.DataFrame([0,1,2,1,1,0]))
d = tf.reshape(b, shape=[-1, 1])
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(b), b.shape)
    print('='*10)
    print(sess.run(d),d.shape)
    --->[[[-0.00229059]]
    --->
    ---> [[ 0.00654686]]
    --->
    ---> [[-0.01635333]]
    --->
    ---> [[ 0.00654686]]
    --->
    ---> [[ 0.00654686]]
    --->
    ---> [[-0.00229059]]] (6, 1, 1)
    --->==========
    --->[[-0.00229059]
    ---> [ 0.00654686]
    ---> [-0.01635333]
    ---> [ 0.00654686]
    ---> [ 0.00654686]
    ---> [-0.00229059]] (6, 1)



7. reduce_mean, reduce_sum

a = tf.Variable(tf.random.normal([3, 2], 0.0, 1), name='v1')
b = tf.nn.embedding_lookup(a, [0,1,2])
c = tf.reduce_mean(b, axis=1)
d = tf.reduce_mean(b, axis=0)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(a),a.shape)
    print('='*10)
    print(sess.run(b),b.shape)
    print('='*10)
    print(sess.run(c),c.shape)
    print('='*10)
    print(sess.run(d),d.shape)
    ---> [[-0.89105785 -1.6476023 ]
    --->  [-2.2250419  -0.31689882]
    --->  [-0.08514699  0.43569314]] (3, 2)
    ---> ==========
    ---> [[-0.89105785 -1.6476023 ]
    --->  [-2.2250419  -0.31689882]
    --->  [-0.08514699  0.43569314]] (3, 2)
    ---> ==========
    ---> [-1.26933    -1.2709703   0.17527308] (3,)
    ---> ==========
    ---> [-1.0670823  -0.50960267] (2,)




a = tf.Variable(tf.random.normal([3, 2], 0.0, 1), name='v1')
b = tf.nn.embedding_lookup(a, [0,1,2])
c = tf.reduce_sum(b, axis=1)
d = tf.reduce_sum(b, axis=0)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(a),a.shape)
    print('='*10)
    print(sess.run(b),b.shape)
    print('='*10)
    print(sess.run(c),c.shape)
    print('='*10)
    print(sess.run(d),d.shape)
    ---> [[ 0.16288437  0.8174733 ]
    --->  [-1.0474529  -1.5918422 ]
    --->  [-0.07028998 -0.09271429]] (3, 2)
    ---> ==========
    ---> [[ 0.16288437  0.8174733 ]
    --->  [-1.0474529  -1.5918422 ]
    --->  [-0.07028998 -0.09271429]] (3, 2)
    ---> ==========
    ---> [ 0.98035765 -2.639295   -0.16300428] (3,)
    ---> ==========
    --->[-0.95485854 -0.8670832 ] (2,)



a = tf.Variable(tf.random.normal([3, 2], 0.0, 1), name='v1')
b = tf.nn.embedding_lookup(a, pd.DataFrame([0,1,2]))
c = tf.reduce_sum(b, axis=1)
d = tf.reduce_sum(b, axis=0)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(a),a.shape)
    print('='*10)
    print(sess.run(b),b.shape)
    print('='*10)
    print(sess.run(c),c.shape)
    print('='*10)
    print(sess.run(d),d.shape)
    ---> [[ 2.0580308   1.643949  ]
    --->  [ 0.70297015  0.05782438]
    --->  [-1.3467138   0.08757292]] (3, 2)
    ---> ==========
    ---> [[[ 2.0580308   1.643949  ]]
    ---> 
    --->  [[ 0.70297015  0.05782438]]
    ---> 
    --->  [[-1.3467138   0.08757292]]] (3, 1, 2)
    ---> ==========
    ---> [[ 2.0580308   1.643949  ]
    --->  [ 0.70297015  0.05782438]
    --->  [-1.3467138   0.08757292]] (3, 2)
    ---> ==========
    ---> [[1.4142873 1.7893463]] (1, 2)


a = tf.Variable(tf.random.normal([3, 1], 0.0, 1), name='v1')
b = tf.nn.embedding_lookup(a, pd.DataFrame([0,1,2]))
c = tf.reduce_sum(b, axis=1)
d = tf.reduce_sum(b, axis=0)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(a),a.shape)
    print('='*10)
    print(sess.run(b),b.shape)
    print('='*10)
    print(sess.run(c),c.shape)
    print('='*10)
    print(sess.run(d),d.shape)
    ---> [[-0.39443222]
    --->  [ 0.33739206]
    --->  [ 1.4046354 ]] (3, 1)
    ---> ==========
    ---> [[[-0.39443222]]
    ---> 
    --->  [[ 0.33739206]]
    ---> 
    --->  [[ 1.4046354 ]]] (3, 1, 1)
    ---> ==========
    ---> [[-0.39443222]
    --->  [ 0.33739206]
    --->  [ 1.4046354 ]] (3, 1)
    ---> ==========
    ---> [[1.3475952]] (1, 1)


a = tf.Variable(tf.random.normal([3, 2], 0.0, 1), name='v1')
b = pd.DataFrame([[0,1,1],[1,2,2]])
c = tf.nn.embedding_lookup(a, b)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(a),a.shape)
    print('='*10)
    print(b)
    print('='*10)
    print(sess.run(c), c.shape)

----->[[-1.3686281  -0.07836375]
-----> [ 0.20003815  1.1355464 ]
-----> [-1.5542912  -2.071774  ]] (3, 2)
----->==========
----->   0  1  2
----->0  0  1  1
----->1  1  2  2
----->==========
----->[[[-1.3686281  -0.07836375]
----->  [ 0.20003815  1.1355464 ]
----->  [ 0.20003815  1.1355464 ]]
----->
-----> [[ 0.20003815  1.1355464 ]
----->  [-1.5542912  -2.071774  ]
----->  [-1.5542912  -2.071774  ]]] (2, 3, 2)
----->    
    
    
    
    
    
    
    
    

8. concat

t1 = np.array([[1, 2, 3], [4, 5, 6]])
t2 = np.array([[7, 8, 9], [10, 11, 12]])
out1 = tf.concat([t1, t2], 0)
out2 = tf.concat([t1, t2], -1)

with tf.Session() as sess:
    print(t1.shape, t2.shape)
    sess.run(tf.initialize_all_variables())
    print(sess.run(out1), out1.shape)
    print('='*10)
    print(sess.run(out2), out2.shape)
    ---> (2, 3) (2, 3)
    ---> [[ 1  2  3]
    --->  [ 4  5  6]
    --->  [ 7  8  9]
    --->  [10 11 12]] (4, 3)
    ---> ==========
    ---> [[ 1  2  3  7  8  9]
    --->  [ 4  5  6 10 11 12]] (2, 6)



t = []
t.append(t1)
out = tf.concat(t, -1)
with tf.Session() as sess:
    print(t)
    sess.run(tf.initialize_all_variables())
    print(sess.run(out), out.shape)

    ---> [array([[1, 2, 3],
    --->        [4, 5, 6]])]
    ---> [[1 2 3]
    --->  [4 5 6]] (2, 3)




9. tf.layers.dense

    tf.layers.dense(
            inputs,  #输入该网络层的数据
            units,   #输出的维度大小，改变inputs的最后一维
            activation, #激活函数 如：activation=tf.nn.relu
            use_bias=True,  # 是否使用偏置项
            kernel_initializer=None, ##初始化器
            bias_initializer=tf.zeros_initializer(), ##偏置项的初始化器，默认初始化为0
            kernel_regularizer=None, ##正则化，可选
            bias_regularizer=None, ##偏置项的正则化，可选
            activity_regularizer=None, ##输出的正则化函数
            kernel_constraint=None,
            bias_constraint=None,
            trainable=True,  # trainable=True:表明该层的参数是否参与训练。如果为真则变量加入到图集合中
            name=None, ##层的名字
            reuse=None ##是否重复使用参数
            )


10. tf.contrib.layers.xavier_initializer

    tf.contrib.layers.xavier_initializer(
     uniform=True,
     seed=None,
     dtype=tf.float32
        )
    返回初始化权重矩阵

    uniform: 使用uniform或者normal分布来随机初始化
    seed: 可以认为是用来生成随机数的seed 
    dtype: 只支持浮点数


11. 计算L1,L2正则化
tf.contrib.layers.l1_regularizer(scale)
tf.contrib.layers.l2_regularizer(scale)


weights = tf.constant([[1.0, -2.0],[-3.0 , 4.0]])
with tf.Session() as sess:
    # 输出为(|1|+|-2|＋|-3|+|4|) x 0.05=0.5。其中0.5为正则化项的权重。
    print(sess.run(tf.contrib.layers.l1_regularizer(0.05)(weights)))

    # 输出为((1)^2+(-2)^2＋(-3)^2+(4)^2)/2 x 0.01=0.15。其中0.5为正则化项的权重。
    # TensorFlow会将L2正则化损失值除以2使得求导得到的结果更加简洁 。
    print(sess.run(tf.contrib.layers.l2_regularizer(0.01)(weights)))

    --->0.5
    --->0.14999999



tf.nn.l2_loss函数

w = tf.constant([[1.0, -2.0],[-3.0 , 4.0]])
with tf.Session() as sess:
    print(sess.run(tf.nn.l2_loss(w)))

--->15.0


12. tf.get_collection(key, scope=None)

该函数可以用来获取key集合中的所有元素，返回一个列表。列表的顺序依变量放入集合中的先后而定。
scope为可选参数，表示的是名称空间（名称域），如果指定就返回名称域中所有放入‘key’的变量的列表，不指定则返回所有变量。


v1 = tf.get_variable(name='v1', shape=[1], initializer=tf.constant_initializer(0))
tf.add_to_collection('loss', v1)
v2 = tf.get_variable(name='v2', shape=[1], initializer=tf.constant_initializer(2))
tf.add_to_collection('loss', v2)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(tf.get_collection('loss','v1'))
    print(tf.get_collection('loss','v2'))
    print(tf.get_collection('loss'))
    print(sess.run(tf.add_n(tf.get_collection('loss'))))

    --->[<tf.Variable 'v1:0' shape=(1,) dtype=float32_ref>]
    --->[<tf.Variable 'v2:0' shape=(1,) dtype=float32_ref>]
    --->[<tf.Variable 'v1:0' shape=(1,) dtype=float32_ref>, <tf.Variable 'v2:0' shape=(1,) dtype=float32_ref>]
    --->[2.]


13. tf.multiply

a = np.array([[1],[2],[3]])
b = np.array([[1,2,3]])
c = tf.multiply(a,b)

with tf.Session() as sess:
    print(sess.run(c),c.shape)
    ---> [[1 2 3]
    ---> [2 4 6]
    ---> [3 6 9]] (3, 3)



14. tf.greater

a = [1.0,0.1,0.2,0.6,0.8]
b = 0.1

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(tf.greater(a, 0.5)))
    print(sess.run(tf.greater(b, 0.5)))
    print('='*10)
    print(sess.run(tf.cast(tf.greater(a, 0.5), tf.int32)))
    print(sess.run(tf.cast(tf.greater(b, 0.5), tf.int32)))
    print('='*10)
    print(sess.run(tf.equal(tf.cast(tf.greater(a, 0.5), tf.int32), 1)))
    print(sess.run(tf.equal(tf.cast(tf.greater(b, 0.5), tf.int32), 1)))
    print('='*10)
    print(sess.run(tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.greater(a, 0.5), tf.int32), 1), tf.float32))))

    --->[ True False False  True  True]
    --->False
    --->==========
    --->[1 0 0 1 1]
    --->0
    --->==========
    --->[ True False False  True  True]
    --->False
    --->==========
    --->0.6 


15. tf.argmax
argmax 指求最大值的索引

import tensorflow as tf
b = [[3,8,9],[2,15,0]]
a = tf.argmax(b,0)
c = tf.argmax(b,1)
with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(c))

---> [0 1 0]
---> [2 1]
   
   
   
16. sparse_placeholder
# 系数矩阵用sparse_placeholder


