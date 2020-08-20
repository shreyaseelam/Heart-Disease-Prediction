#import random
import numpy as np
#import math
import matplotlib.pyplot as plt
from sklearn import preprocessing
#import pandas
import tensorflow as tf
#extracting data
def get_data():
        data = []
        ds = [] 
        lines = [line.rstrip('\n') for line in open('cleavland.txt')]
        l = len(lines) 
        
        for i in range(l):
            data.append(lines[i].split('\t'))
        m = len(data[0])
        #print(m)
        for x in data:
            line = []
            for i in range(0,m):
                line.append(float(x[i]))
            ds.append(line)

        return l,m-1,ds
NoTr,NoF,data = get_data()
#print(NoTr,NoF,data)
tempdata = list(data)
tempdata = list(data)
#print(NoD)
#splitting the data into training, validation and testing
NoTe = 31       #956
#print(NoTe)
Te = [] # Test samples
I = []
#print("out")
for i in range(int(NoTe)):
    #print("x")
    Te.append(data[i])
    I.append(i)
#print(len(np.array(Te)))
#print(I)
to_del = object()
for index in I:
    tempdata[index] = to_del
for _ in I:
    tempdata.remove(to_del)
#print(np.array(tempdata))
Vd = list(tempdata) #Validation samples
#print(len(Vd))
NoTr =  222  #6698
Tr = [] # Train samples
I = np.random.choice(len(tempdata)-1, NoTr , replace=False)
#print(I)
for j in range(len(I)):
    Tr.append(tempdata[I[j]])

to_del = object()

for index in I:
    Vd[index] = to_del

for _ in I:
    Vd.remove(to_del)
  
########################
#scaling the input
#print(np.array(Tr))
#print(np.array(Te))
#print(np.array(Vd))

#print(len(np.array(Tr)))
#print(len(np.array(Te)))
#print(len(np.array(Vd)))

Tr_scaled = preprocessing.scale(Tr)
Te_scaled = preprocessing.scale(Te)
Vd_scaled = preprocessing.scale(Vd)

Tr_scaled = np.array(Tr_scaled)
Te_scaled = np.array(Te_scaled)
Vd_scaled = np.array(Vd_scaled)

#print(Tr_scaled)
#print(Te_scaled)
#print(Vd_scaled)
#Splitting all the datasets in to x coordinated and y coordinated
Tr_last  =  Tr_scaled[:,13]
Tr = np.delete(Tr_scaled, -1, axis=1) #First 13 column's 
Tr = np.array(Tr)
Tr_last = Tr_last.reshape(-1,1) #Last column 
Tr_last = np.array(Tr_last)

Te_last  =  Te_scaled[:,13]
Te = np.delete(Te_scaled, -1, axis=1) #First 13 column's 
Te = np.array(Te)
Te_last = Te_last.reshape(-1,1) #Last column 
Te_last = np.array(Te_last)

Vd_last  =  Vd_scaled[:,13]
Vd = np.delete(Vd_scaled, -1, axis=1) #First 13 column's 
Vd = np.array(Vd)
Vd_last = Vd_last.reshape(-1,1) #Last column 
Vd_last = np.array(Vd_last)

train_list = []
test_list = []
validation_list = []
train_list_y = []
test_list_y = []
validation_list_y = []
#Initialize the parameters
n_input = 13
#n_hidden1 = 20
#n_hidden2 = 10
n_hidden = 30
n_output = 1
learning_rate = 0.001
epochs = 200
#Initalizing the X and Y 
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#W1 = tf.Variable(tf.random_uniform([n_input,n_hidden1],-1.0,1.0))
#W2 = tf.Variable(tf.random_uniform([n_hidden1,n_hidden2],-1.0,1.0))
#W3 = tf.Variable(tf.random_uniform([n_hidden2,n_output],-1.0,1.0))
#b1 = tf.Variable(tf.zeros([n_hidden1]),name ="Bias1")
#b3 = tf.Variable(tf.zeros([n_hidden2]),name ="Bias1")
#b2 = tf.Variable(tf.zeros([n_output]),name ="Bias2")

W1 = tf.Variable(tf.random_uniform([n_input,n_hidden],-1.0,1.0))
#W2 = tf.Variable(tf.random_uniform([n_hidden1,n_hidden2],-1.0,1.0))
W2 = tf.Variable(tf.random_uniform([n_hidden,n_output],-1.0,1.0))
b1 = tf.Variable(tf.zeros([n_hidden]),name ="Bias1")
#b3 = tf.Variable(tf.zeros([n_hidden2]),name ="Bias1")
b2 = tf.Variable(tf.zeros([n_output]),name ="Bias2")

L2 = tf.sigmoid(tf.matmul(X,W1)+ b1)
#L3 = tf.sigmoid(tf.matmul(L2,W2)+ b3)
y = tf.sigmoid(tf.matmul(L2,W2)+ b2)
print(str(y))

#error = tf.reduce_mean(-Y*tf.log(y) - (1-Y)*tf.log(1-y))
error = tf.reduce_sum(y-Y) #Sigmoid
#error = tf.reduce_mean(y-Y) #relu,tanh
#error = (y-Y)**2
#error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)

init = tf.initialize_all_variables()

with tf.Session()as session:
    session.run(init)
    for i in range(epochs):
        session.run(optimizer,feed_dict = {X:Tr,Y:Tr_last})
        #for l in range(len(Tr)):
        train_list.append(session.run(error,feed_dict = {X:Tr,Y:Tr_last}))
        
        #for m in range(len(Te)):
        test_list.append(session.run(error,feed_dict = {X:Te,Y:Te_last}))
        #for n in range(len(Vd)):
        validation_list.append(session.run(error,feed_dict = {X:Vd,Y:Vd_last}))

  
for i in range(1,200+1):
    train_list_y.append(i)
    test_list_y.append(i)
    validation_list_y.append(i)
tf.Print(test_list_y,[test_list_y])


#output = scalery.inverse_transform(test_list)
#print(output)
#print(train_list)
#print(train_list_y)'''
plt.plot(train_list_y,train_list,'y')
#plt.show()
plt.plot(test_list_y,test_list,'r')
#plt.show()
plt.plot(validation_list_y,validation_list,'m')
plt.show()

        
           
