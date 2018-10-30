import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#load te fashion-mnist pre shuffled train data and test data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()# Print training set shape - note there are 60,000 training data of image size of 28x28, 60,000 train labels)
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

# Print the number of training and test datasets
print(x_train.shape[0], 'train set')
print(x_test.shape[0], 'test set')

# Define the text labels
fashion_mnist_labels = ["T-shirt/top",  # index 0
                        "Trouser",      # index 1
                        "Pullover",     # index 2
                        "Dress",        # index 3
                        "Coat",         # index 4
                        "Sandal",       # index 5
                        "Shirt",        # index 6
                        "Sneaker",      # index 7
                        "Bag",          # index 8
                        "Ankle boot"]   # index 9

# Image index, between 0 and 59,999
img_index = 5
# y_train contains the lables, ranging from 0 to 9
label_index = y_train[img_index]
# Print the label, for example 2 Pullover
print ("y = " + str(label_index) + " " +(fashion_mnist_labels[label_index]))
# # Show one of the images from the training dataset
plt.imshow(x_train[img_index])
plt.show()


#-------------------data normalization--------------------#

x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

print("Number of train data - " + str(len(x_train)))
print("Number of test data - " + str(len(x_test)))



#--------------split data into train/validation/test data sets---------#
# training data- used for training the model
# validation data -  used for tuning the hyperparameners and evaluate the models
# test data - used to test the model after the model has gone through initual
    # vetting by the validation set

# further break training data into train/validation sets(#put 5000 into
    #validation set and keep remaining 55,000 for train)
(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

# # reshape input data from (28, 28) to (28, 28, 1)
w, h = 28, 28
x_train = x_train.reshape(x_train.shape[0], w, h, 1)
x_valid = x_valid.reshape(x_valid.shape[0], w, h, 1)
x_test = x_test.reshape(x_test.shape[0], w, h, 1)

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
        #converts a class vector(integer) to binary class matrix
        # arguments:
            # y: class vector to be converted into a matrix(integers from 0 to num_classes)
            # num_classes: total number of classes
y_valid = tf.keras.utils.to_categorical(y_valid, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# print training set shape
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

#print the number of training, validation, and test datasets
print(x_train.shape[0], 'train set')
print(x_valid.shape[0], 'validation set')
print(x_test.shape[0], 'test set')

model = tf.keras.Sequential()

#must define the input shape in the first layer of the neural network
model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 2, padding = "same", activation = "relu", input_shape = (28,28,1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size = 2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 2, padding = "same", activation = "relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size = 2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation = "relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation = "softmax"))

#take a look at model summary
model.summary()
