# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import KFold, train_test_split, cross_val_score, cross_validate
from sklearn import svm, tree



iris = datasets.load_iris()
X = pd.DataFrame(iris['data'], columns=iris['feature_names'])

y = pd.Series(iris['target_names'][iris['target']])
# y = pd.get_dummies(y)

#线性核SVM
linear_svm = svm.SVC(C=1, kernel='linear')
linear_scores = cross_validate(linear_svm, X, y, cv=5, scoring='accuracy')

linear_scores['test_score'].mean()

0.98000000000000009

#高斯核SVM
rbf_svm = svm.SVC(C=1)
rbf_scores = cross_validate(rbf_svm, X, y, cv=5, scoring='accuracy')

rbf_scores['test_score'].mean()

0.98000000000000009

# 一层隐藏层的BP神经网络，神经元个数为16
x_input = tf.placeholder('float', shape=[None, 4])
y_input = tf.placeholder('float', shape=[None, 3])

keep_prob = tf.placeholder('float', name='keep_prob')

W1 = tf.get_variable('W1', [4, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
b1 = tf.get_variable('b1', [16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

h1 = tf.nn.relu(tf.matmul(x_input, W1) + b1)
h1_dropout = tf.nn.dropout(h1, keep_prob=keep_prob, name='h1_dropout')

W2 = tf.get_variable('W2', [16, 3], initializer=tf.contrib.layers.xavier_initializer(seed=0))
b2 = tf.get_variable('b2', [3], initializer=tf.contrib.layers.xavier_initializer(seed=0))

y_output = tf.matmul(h1_dropout, W2) + b2

# 定义训练步骤、准确率等
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_output, labels=y_input))

train_step = tf.train.AdamOptimizer(0.003).minimize(cost)

correct_prediction = tf.equal(tf.argmax(y_output, 1), tf.argmax(y_input, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# 将目标值one-hot编码
y_dummies = pd.get_dummies(y)

sess = tf.Session()
init = tf.global_variables_initializer()
costs = []
accuracys = []

for train, test in KFold(5, shuffle=True).split(X):
    sess.run(init)
    X_train = X.iloc[train, :]
    y_train = y_dummies.iloc[train, :]
    X_test = X.iloc[test, :]
    y_test = y_dummies.iloc[test, :]

    for i in range(1000):
        sess.run(train_step, feed_dict={x_input: X_train, y_input: y_train, keep_prob: 0.3})

    test_cost_, test_accuracy_ = sess.run([cost, accuracy],
                                          feed_dict={x_input: X_test, y_input: y_test, keep_prob: 1})
    accuracys.append(test_accuracy_)
    costs.append(test_cost_)

print(accuracys)
print(np.mean(accuracys))
