import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#function that changes the classes into one-hot vectors
def toOneHot(word):
    if(word == 'Iris-setosa'):
        return np.array([1,0,0])
    elif(word == 'Iris-versicolor'):
        return np.array([0,1,0])
    else:
        return np.array([0,0,1])
    
#function that takes logits (or one Hot vectors) and return the appropiate class
def toWord(array):
    if(array[0] > array[1] and array[0] > array[2]):
        return 'Iris-setosa'
    if(array[1] > array[0] and array[1] > array[2]):
        return 'Iris-versicolor'
    if(array[2] > array[0] and array[2] > array[1]):
        return 'Iris-virginica'
    

#loading in our data
iris_data = pd.read_csv("iris_data.csv")
        
#Creating a model with an hidden elu layer of 100 neurons
#and softmax ouput layer of 3 to give us the predicted probability for each class

#used categorical cross entropy because we are using one hot vectors

#used Adam optimzer because it is good in general
model = tf.keras.Sequential([tf.keras.layers.Dense(100, activation='elu'),
                             tf.keras.layers.Dense(3, activation='softmax')])
loss_fn = tf.keras.losses.CategoricalCrossentropy()
op = tf.keras.optimizers.Adam()

#seperate the features from the labels and changes classes into one hot vectors
features = iris_data.copy()
labels = features.pop('class')
oneHots = list(labels.map(toOneHot))
oneHots = np.array(list(map(tf.constant, oneHots)))

#array to store accuracies for each iteration of our loop
acc_arr = []


for i in range(100):
    #split data into 80-20 training testing split
    x_train,x_test,y_train,y_test=train_test_split(features, oneHots, test_size=0.2)

    #train the model using our training data
    model.compile(loss = loss_fn, optimizer = op, metrics=['accuracy'])
    model.fit(x_train, y_train, verbose = 0, epochs=50)

    #find predictions for our testing data
    y_hat = list(map(toWord, model.predict(x_test)))
    y = list(map(toWord, y_test))

    #find the accuracy for our model on testing data and add it to our array    
    acc = accuracy_score(y, y_hat)
    acc_arr.append(acc)

#find the average accuracy from our iterations
mean_acc = np.mean(acc_arr)
    

print('Avg accuracy (testing data) for 100 iterations using 80-20, training-testing split for data : ')
print(mean_acc)




