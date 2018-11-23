import tensorflow as tf
import numpy as np
import csv
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
comment_train=[]
comment_test=[]

f=open("../hw1_data/cancer_train.csv","r")
read_train=csv.reader(f)

ft=open("../hw1_data/cancer_train.csv","r")
read_test=csv.reader(ft)

for line in read_train:
    comment_train.append(line)

for line in read_test:
    comment_test.append(line)

f.close()
ft.close()

#separate data and label
test=np.array(comment_test)
x_data=test[0:,1:]
y_data=test[0:,0]
onehot_encoder = OneHotEncoder(sparse=False)
y_data = y_data.reshape(len(y_data), 1)
y_data = onehot_encoder.fit_transform(y_data)


with tf.Session() as sess:
    X = tf.placeholder(tf.float32)
   

    saver=tf.train.import_meta_graph('saved/my_model'+'.meta')
    saver.restore(sess, tf.train.latest_checkpoint('saved'))

    graph = tf.get_default_graph()

    hypothesis = x_data

    score = sess.run(hypothesis, feed_dict={X:x_data})
    print(score)
