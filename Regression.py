import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Reading the dataset from the file
coeff = []
dataset = pd.read_csv('cleavland.csv')
my_list =list(dataset)
#print(dataset.head())
#dataset.describe()
#Split the data set into independent and dependent varialbles
X = dataset[[my_list[1],my_list[2],my_list[3],my_list[4],my_list[5],my_list[6],my_list[7],my_list[8],my_list[9],my_list[10],my_list[11],my_list[12],my_list[13]]]
y = dataset[my_list[-1]]
coeff.append(my_list[0])
#Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#Training the regression model 
regressor = LinearRegression()  
regressor.fit(X_train, y_train)
#Predecting the output from the model
y_pred = regressor.predict(X_test)
pred = []
output = []
for i in range(61):
    m = y_pred.flat[i]
    pred.append(m)
    if m > 0.5:
        output.append(1)
    else:
        output.append(0)
print(output)
#Evaluating the model
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,output))  
print(classification_report(y_test,output))
    
#df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
#print(my_matrix)     
#print(df)  
#print(my_matrix[0][0])
