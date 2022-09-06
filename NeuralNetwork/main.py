from matplotlib import pyplot as plt
from numpy import argmax, load, array, invert, newaxis
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.losses import sparse_categorical_crossentropy
from tensorflow import nn
import cv2

data_path = "/home/ayoubamer/Workspace/ML/MachineLearning/Datasets/mnist.npz"

# load handwrite digits data from tensorflow
# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(
#     path=data_path
# )

# if you already have the mnist data

data = load(data_path)

x_train = normalize(data["x_train"], axis=1)
x_test = normalize(data["x_test"], axis=1)
y_train = data["y_train"]
y_test = data["y_test"]

# create feed ward neural network
model = Sequential()
# Input Layer
model.add(Flatten(input_shape=(28, 28)))
# First Hidden Layer
# Dense means all neurons are connected to prev and next neuron
# units are number of neurons in that layer; activation is the action to take by every neuron
model.add(Dense(units=130, activation="relu"))
# Second Hidden Layer
model.add(Dense(units=130, activation="relu"))
# Output Layer
model.add(Dense(units=10, activation="softmax"))

# compiule model
model.compile(optimizer="Adam",
              loss=sparse_categorical_crossentropy, metrics=["accuracy"])

model.fit(x_train, y_train, epochs=3)

# evaluating model
accuracy, loss = model.evaluate(x_test, y_test)

print(accuracy)
print(loss)

# save model
model.save("DigitsModel")
print("saved")

# test model
img = cv2.imread("Images/5.png")[:, :, 0]
img = invert(array([img]))  # digit color is the black
prediction = model.predict(img)
print(argmax(prediction))
# add cmp because we need the image in black and white so neural network can work with it
plt.imshow(img[0], cmap=plt.cm.binary)
plt.show()
