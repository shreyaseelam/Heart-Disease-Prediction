import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier 
coeff = []
dataset = pd.read_csv('cleavland.csv')
my_list =list(dataset)
#print(dataset.head())
#dataset.describe()
#spliting the input and output
X = dataset[[my_list[1],my_list[2],my_list[3],my_list[4],my_list[5],my_list[6],my_list[7],my_list[8],my_list[9],my_list[10],my_list[11],my_list[12],my_list[13]]]
y = dataset[my_list[-1]]
#print(y.head())
coeff.append(my_list[0])
#split the dataset into trainig and testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#Normalizing the set to make the model learn properly 
scaler = StandardScaler()  
scaler.fit(X_train)
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)
#print("reached here")
#Building the model with three input layers and 20 neurons each
mlp = MLPClassifier(hidden_layer_sizes=(20, 20, 20), max_iter=1000)  
mlp.fit(X_train, y_train.values.ravel())
#predicting the output for the test data from the model
predictions = mlp.predict(X_test)
pred = predictions.tolist()
#print(predictions)
print(pred)
#Evaluation of model
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))
#Visualisation
output = []
xcord = []
for j in range(61):
    xcord.append(j)

for i in pred:
    
    if i>0.5:
        output.append(1)
    else:
        output.append(0)
    
plt.scatter(xcord,output,'r')
plt.show()
