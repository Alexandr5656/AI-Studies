import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(mx_train, my_train), (mx_test, my_test) = keras.datasets.mnist.load_data()
(fx_train, fy_train), (fx_test, fy_test) = keras.datasets.fashion_mnist.load_data()

# Scale images to the [0, 1] range
mx_train = mx_train.astype("float32") / 255
mx_test = mx_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
mx_train = np.expand_dims(mx_train, -1)
mx_test = np.expand_dims(mx_test, -1)
print("x_train shape:", mx_train.shape)
print(mx_train.shape[0], "train samples")
print(mx_test.shape[0], "test samples")


# convert class vectors to binary class matrices
my_train = keras.utils.to_categorical(my_train, num_classes)
my_test = keras.utils.to_categorical(my_test, num_classes)
#
#
#model = keras.Sequential(
#    [
#        keras.Input(shape=input_shape),
#        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
#        layers.MaxPooling2D(pool_size=(2, 2)),
#        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
#        layers.MaxPooling2D(pool_size=(2, 2)),
#        layers.Flatten(),
#        layers.Dropout(0.5),
#        layers.Dense(num_classes, activation="softmax"),
#    ]
#)
#
#model.summary()
#
#batch_size = 128
#epochs = 15
#
#model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
#
#model.fit(mx_train, my_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
#
#
#
#
#score = model.evaluate(mx_test, my_test, verbose=0)
model = keras.models.load_model('model_weights.h5')

# Scale images to the [0, 1] range
fx_train = fx_train.astype("float32") / 255
fx_test = fx_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
fx_train = np.expand_dims(fx_train, -1)
fx_test = np.expand_dims(fx_test, -1)
print("x_train shape:", fx_train.shape)
print(fx_train.shape[0], "train samples")
print(fx_test.shape[0], "test samples")


# convert class vectors to binary class matrices
fy_train = keras.utils.to_categorical(fy_train, num_classes)
fy_test = keras.utils.to_categorical(fy_test, num_classes)

score = model.evaluate(fx_test, fy_test, verbose=0)


print("fTest loss:", score[0])
print("fTest accuracy:", score[1])

#TODO Save model
#TODO Takout out front end and back end