import tensorflow as tf 
import pandas as pd 
from pandas import DataFrame as DF,Series
import numpy as np
import gc


data = pd.read_csv('train_mycode-multilayer.csv')

#filter only needed features
data = data.fillna({'Age':-1,'Cabin':'Unk','Embarked':'Unk','Fare':-1,'SibSp':0,'Parch':0})

#transform features
data.loc[:,'Sex'] = (data.Sex == 'female').astype(int)

#data split
Xtr = data.loc[:,['Pclass','Sex','Age','SibSp','Parch','Fare']].sample(frac=0.75)
Xts = data.loc[~data.index.isin(Xtr.index)].loc[:,['Pclass','Sex','Age','SibSp','Parch','Fare']]

Ytr = pd.get_dummies(data[data.index.isin(Xtr.index)].Survived).values
Yts = pd.get_dummies(data[~data.index.isin(Xtr.index)].Survived).values


#hyper parameters
learning_rate = 0.01
n_input = Xtr.shape[1]
n_nodes_1 = 32
n_nodes_2 = 32
n_output = Ytr.shape[1]

batch_size = 100

X = tf.placeholder('float',[None,n_input])
Y = tf.placeholder('float',[None,n_output])

def create_weights(shape):
	initializer = tf.random_normal(shape,stddev=0.1)
	W = tf.Variable(initializer)
	return W

def create_biases(shape):
	initializer = tf.random_normal(shape)
	B = tf.Variable(initializer)
	return B 

#assign the weights we want for this model
weights = {'w1': create_weights([n_input, n_nodes_1]),  # first and only hidden layer weights
           'w2': create_weights([n_nodes_1, n_nodes_2]),
           'w_out': create_weights([n_nodes_2, n_output])}  # output layer weights

# assign biases
biases = {'b1': create_biases([n_nodes_1]),
          'b2': create_biases([n_nodes_2]),
          'b_out': create_biases([n_output])}

# hidden layer
z1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])  # argument for the activation function
a1 = tf.nn.sigmoid(z1)  # the activation function

z2 = tf.add(tf.matmul(a1, weights['w2']), biases['b2'])  # argument for the activation function
a2 = tf.nn.sigmoid(z2)  # the activation function

# output layer
logits = tf.add(tf.matmul(a2, weights['w_out']), biases['b_out'])  # operates on previous layer outputs
# yhat = tf.nn.softmax(logits)  # gives class probabilities


# Back propogation
# define loss function
# RMSE
loss = tf.sqrt(tf.reduce_mean(tf.square(Y-logits)))

# define optimizer
# optimize = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
optimize = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# train over 300 epochs
n_epochs = 300
for epoch in range(1, n_epochs + 1):
    # train on one batch at a time
    for i in range(0, len(Xtr), batch_size):
        sess.run(optimize, feed_dict={X: Xtr[i: i+batch_size],
                                      Y: Ytr[i: i+batch_size]})
    #print(Xtr);
    # compute training loss for printing progress
    if (epoch%10 == 0) | (epoch == 1):
        loss_tr = sess.run(loss, feed_dict={X: Xtr, Y: Ytr})
        print('Epoch {}, loss: {:.3f}'.format(epoch, loss_tr, 3))
        

# compute loss for test data
loss_ts = sess.run(loss, feed_dict={X: Xts, Y: Yts})
print(10*'-')
print('Test loss: {:.3f}'.format(np.round(loss_ts, 3)))

del data 
gc.collect()
