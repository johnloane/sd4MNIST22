import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import random

np.random.seed(0)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
assert(X_train.shape[0] == y_train.shape[0]), "The number of images and labels does not match for the training set"
assert(X_test.shape[0] == y_test.shape[0]), "The number of images and labels does not match for the test set"
assert(X_train.shape[1:] == (28, 28))
assert(X_test.shape[1:] == (28, 28))

num_of_samples = []
cols = 5
num_classes = 10
fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 10))
fig.tight_layout()
for i in range(cols):
    for j in range(num_classes):
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected)-1),:,:], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i==2:
            axs[j][i].set_title(str(j))
            num_of_samples.append(len(x_selected))

print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of training set")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()
np.set_printoptions(threshold=np. inf)
print(y_train)

# One hot encode training and test labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
print(y_train)




