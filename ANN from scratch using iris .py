import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import warnings

np.random.seed(42)

iris = datasets.load_iris()

inputs = iris.data 
outputs = iris.target 

def to_one_hot(Y):
    n_col = np.amax(Y) + 1
    binarized = np.zeros((len(Y), n_col))
    for i in range(len(Y)):
        binarized[i, Y[i]] = 1.
    return binarized

#sigmoid and its derivative
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x)*(1 - sigmoid(x))

#Normalize array
def normalize(X, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)

#2nd check
# iris = pd.read_csv('/content/Iris.csv')
# iris['Species'].replace(['Iris-setosa', 'Iris-virginica', 'Iris-versicolor'], [0, 1, 2], inplace=True)

xx = normalize(inputs)
yy = to_one_hot(outputs)

X_train, X_test, y_train, y_test = train_test_split(xx, yy, test_size=0.33 , shuffle=True)
    
#Weights
w0 = 2*np.random.random((4, 5)) - 1 #for input   - 4 inputs, 3 outputs
w1 = 2*np.random.random((5, 3)) - 1 #for layer 1 - 5 inputs, 3 outputs

n = 0.1 #learning rate

#Errors - for graph later
errors = []

for i in range(10000):

    #Feed forward
    layer0 = X_train
    layer1 = sigmoid(np.dot(layer0, w0))
    layer2 = sigmoid(np.dot(layer1, w1))

    #Back propagation using gradient descent
    layer2_error = y_train - layer2
    layer2_delta = layer2_error * sigmoid_deriv(layer2)
    
    layer1_error = layer2_delta.dot(w1.T)
    layer1_delta = layer1_error * sigmoid_deriv(layer1)
    
    w1 += layer1.T.dot(layer2_delta) * n
    w0 += layer0.T.dot(layer1_delta) * n
    error = np.mean(np.abs(layer2_error))
    errors.append(error)
    accuracy = (1 - error) * 100

print("Training Accuracy " + str(round(accuracy,2)) + "%")

real_name = []

#Validate
layer0 = X_test
layer0 = normalize(layer0) # use only when there is one value
layer1 = sigmoid(np.dot(layer0, w0))
layer2 = sigmoid(np.dot(layer1, w1))

yp = np.argmax(layer2,axis=1) # prediction
# yp = np.argmax(layer2)# no need for axis=1 , when there is only one value passed #**$%
print(f"Predicted value  : {yp}")
real_name.append(yp)

#**%$ only use when the data is custom , more than 1 | unnder construction 
dat = np.array(real_name)
data = dat.reshape(dat.shape[1],)
dat = pd.DataFrame(data)
dat['predicted_nuemric'] = pd.DataFrame(data)
dat = dat[['predicted_nuemric']].replace([0,1,2], ['Iris-setosa','Iris-versicolor','Iris-virginica'])
dat
