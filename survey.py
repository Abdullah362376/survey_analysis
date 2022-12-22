import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# tenserflow & keras for the learning part
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# sklearn lib to do some data preprocessing & evaluating matrices only
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import metrics


#upload the data set
dataset=pd.read_csv('D_set.csv')
dataset = dataset.dropna(axis=1)
dataset = dataset.values
#
# # seprating the data into features X and labels Y
X=dataset[:, :-1]
X =X.astype('float32')
Y = dataset[:,-1]
Y = Y.astype('float32')
np.save("label",Y)
#scale data between 0 & 1
x_normed = (X-X.min())/ (X.max()-X.min())
X = x_normed

# zero mean data
mean = X.mean(axis = 0)
X -= mean

# unit stadndard deviation
std = X.std(axis = 0)
X /= std
np.save("data",X)
# split the data into train & test
x_train ,x_test ,y_train ,y_test= train_test_split(X,Y,test_size=0.2)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

# constructing the ANN module
model = Sequential()
model.add(Dense(8,input_dim = len(x_train[0,: ]),activation = 'relu'))
# model.add(Dense(4,activation = 'relu'))
model.add(Dense(1,activation = 'sigmoid'))
#compile the model
model.compile(loss = 'binary_crossentropy', optimizer= 'adam',metrics= ['accuracy'])
# Train the model
model.fit(x = x_train, y = y_train, epochs=50, verbose=1)

# extract prediction of the lables based on your trained model given your features
predect_train = model.predict(x_train).round()
predect_test = model.predict(x_test).round()

# printing the accuracy and other evaluation parameters using sklearn
accuracy_train = accuracy_score(y_train,predect_train.round())
accuracy_test = accuracy_score(y_test,predect_test.round())
precision_train=precision_score(y_train,predect_train.round())
precision_test=precision_score(y_test,predect_test.round())
Sensitivity_recall_train = recall_score(y_train, predect_train)
Sensitivity_recall_test = recall_score(y_test, predect_test)
Specificity_train= metrics.recall_score(y_train, predect_train, pos_label=1)
Specificity_test= metrics.recall_score(y_test, predect_test, pos_label=1)
f1_train = f1_score(y_train,predect_train.round())
f1_test = f1_score(y_test,predect_test.round())
print({"TRAIN:  ""Accuracy":accuracy_train,"Precision":precision_train,"Sensitivity_recall":Sensitivity_recall_train,"Specificity":Specificity_train,"F1_score":f1_train})
print({"TEST:   ""Accuracy":accuracy_test," Precision":precision_test," Sensitivity_recall":Sensitivity_recall_test," Specificity":Specificity_test," F1_score":f1_test})
confusion_matrix_train = metrics.confusion_matrix(y_train, predect_train)
confusion_matrix_test  = metrics.confusion_matrix(y_test, predect_test)

train_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_train, display_labels = [False, True])
test_display  = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_test , display_labels = [False, True])

train_display.plot()
plt.show()
test_display.plot()
plt.show()
