import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot  as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题


'''
客户流失预测模型
owNumber：行号 ×
CustomerID：用户编号 ×
Surname：用户姓名 ×
CreditScore：信用分数
Geography：用户所在国家/地区 ×
Gender：用户性别
Age：年龄
Tenure：当了本银行多少年用户
Balance：存贷款情况
NumOfProducts：使用产品数量
HasCrCard：是否有本行信用卡
IsActiveMember：是否活跃用户
EstimatedSalary：估计收入
Exited：是否已流失，这将作为我们的标签数据
'''


df = pd.read_csv("./data/select-data.csv")          #训练样本集读取
df_test = pd.read_csv("./data/scalar-test.csv")     #测试数据集读取
print("构建向量.........")
# 构建向量
train = []                                          #训练样本集的读取
target = []
for i in range(0, len(df["EstimatedSalary"])):      #循环次数为任意一个字段的行数（这里选择EstimatedSalary）
    mid = []                                        #读条件
    mid.append(df["Geography"][i])
    mid.append(df["Gender"][i])
    mid.append(df["EB"][i])
    mid.append(df["Age"][i])
    mid.append(df["EstimatedSalary"][i])
    mid.append(df["NumOfProducts"][i])
    mid.append(df["CreditScore"][i])
    mid.append(df["Tenure"][i])
    mid.append(df["HasCrCard"][i])
    mid.append(df["IsActiveMember"][i])
    target.append(df["Exited"][i])                  #读标签
    train.append(mid)                               #训练样本的条件
train = np.array(train)
target = np.array(target)

test = []                                           #测试样本集的读取
test_target = []

for i in range(0, len(df_test["EstimatedSalary"])):
    mid = []
    mid.append(df_test["Geography"][i])
    mid.append(df_test["Gender"][i])
    mid.append(df_test["EB"][i])
    mid.append(df_test["Age"][i])
    mid.append(df_test["EstimatedSalary"][i])
    mid.append(df_test["NumOfProducts"][i])
    mid.append(df_test["CreditScore"][i])
    mid.append(df_test["Tenure"][i])
    mid.append(df_test["HasCrCard"][i])
    mid.append(df_test["IsActiveMember"][i])
    test_target.append(df_test["Exited"][i])
    test.append(mid)
test = np.array(test)
test_target = np.array(test_target)
# train = np.trunc(train * 100)

# 随机打乱训练集与标签
train, target = shuffle(train, target)              #打乱训练集样本的顺序
target = target.reshape(-1, 1)                      #规整数据格式
test_target = test_target.reshape(-1, 1)

# One-Hot编码
enc = OneHotEncoder()
enc.fit(test_target)
test_target = enc.transform(test_target).toarray()
enc.fit(target)
target = enc.transform(target).toarray()
enc.fit(test_target)


# 定义输入占位符
x = tf.placeholder(tf.float32, shape=(None, 10))
# # 二分类问题 [0,1]
y = tf.placeholder(tf.float32, shape=(None, 2))
keep = tf.placeholder(tf.float32)

# 定义网络结构
# layer1 第一个隐层
#normal是正态分布，stddev是标准差，truncated是截断的意思（删除影响较大的节点，避免）
var1 = tf.Variable(tf.truncated_normal([10, 256], stddev=0.1))  #大小是10*256，10表示标签个数（也是输入个数），256表示隐藏层个数
bias1 = tf.Variable(tf.zeros([256]))                            #偏置初始化，全部取0
hc1 = tf.add(tf.matmul(x, var1), bias1)                         #输出=输入*权重+偏置
h1 = tf.sigmoid(hc1)                                            
h1 = tf.nn.dropout(h1, keep_prob=keep)                          #为了避免过拟合，采用漏失操作，以一定概率删掉一些节点连接

# layer2
var2 = tf.Variable(tf.truncated_normal([256, 256], stddev=0.1))
bias2 = tf.Variable(tf.zeros([256]))
hc2 = tf.add(tf.matmul(h1, var2), bias2)
h2 = tf.sigmoid(hc2)
h2 = tf.nn.dropout(h2, keep_prob=keep)

# layer3
var3 = tf.Variable(tf.truncated_normal([256, 2], stddev=0.1))
bias3 = tf.Variable(tf.zeros([2]))
hc3 = tf.add(tf.matmul(h2, var3), bias3)
h3 = tf.nn.softmax(hc3)                                         #输出用概率化的激活函数，


# 定义损失
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=h3, labels=y))  #选择cross_entropy作为损失函数
tf.summary.scalar('loss', loss)

# 定义正确率
ac = tf.cast(tf.equal(tf.argmax(h3, 1), tf.argmax(y, 1)), tf.float32)
acc = tf.reduce_mean(ac)                                        #acc为正确率
tf.summary.scalar('accuracy', acc)

# 定义优化器
optimzer = tf.train.AdamOptimizer(1e-3).minimize(loss)          #adam动态学习率优化方法

merge_summary = tf.summary.merge_all()

isTrain = 1

# 定义训练
print("正在训练.....")
saver = tf.train.Saver(max_to_keep=1)

with tf.Session() as sess:
    if isTrain:
        init_op = tf.global_variables_initializer() #初始化
        sess.run(init_op)
        summary_writer = tf.summary.FileWriter('./logs/', sess.graph)
        nepoch=[]
        trainacc=[]
        testacc=[]
        loss1=[]
        for i in range(0, 10001):   #训练一万次
             
            sess.run(optimzer, feed_dict={x: train, y: target, keep: 0.5})
            train_summary = sess.run(merge_summary, feed_dict={x: train, y: target, keep: 1})   #run表示梯度下降法
            summary_writer.add_summary(train_summary, i)
           
            if i % 50 == 0:     #每训练50次，检测准确度
                accu = sess.run(acc, feed_dict={x: train, y: target, keep: 1})          #训练样本准确度
                accuT = sess.run(acc, feed_dict={x: test, y: test_target, keep: 1})     #测试样本准确度
                losss = sess.run(loss, feed_dict={x: train, y: target, keep: 1})        #代价函数
               
                print("epoch:" + str(i) + "   train_acc:" + str(accu) + "   test_acc:" + str(accuT) + "   loss:" + str(
                    losss))
      
                nepoch.append(i)
                trainacc.append(accu)
                testacc.append(accuT)
                loss1.append(losss)
           
        plt.title("BP神经网络训练性能曲线")
        plt.xlabel('训练次数')
        plt.ylabel('训练样本和检验样本的准确度')
        plt.plot(nepoch,trainacc,nepoch,testacc)
        plt.plot(nepoch,trainacc,nepoch,loss1)
        plt.show()

saver.save(sess, './model/bank.ckpt', global_step=i)
'''
    else:
        f = open("./result/NN-target.txt", "w")
        model_file = tf.train.latest_checkpoint('./NN-model/')
        saver.restore(sess, model_file)
        tar = sess.run(h3, feed_dict={x: test, y: test_target, keep: 1})
        tar = sess.run(tf.argmax(tar, 1))
        for i in range(0, len(tar)):
            f.write(str(tar[i]) + " ")
        f.close()
        print(tar)'''
