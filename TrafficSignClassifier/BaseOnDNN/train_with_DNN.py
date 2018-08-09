import pickle
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras.layers import Dense, Activation
from keras import regularizers

# 载入使用数据
training_file = "./Data/train.p"
testing_file = "./Data/test.p"

with open(training_file, mode="rb") as f:
    train = pickle.load(f)
with open(testing_file, mode="rb") as f:
    test = pickle.load(f)

x_train, y_train = train["features"] / 255, train["labels"]
x_test, y_test = test["features"] / 255, test["labels"]

x_train, x_test = x_train.reshape([34799, 32 * 32 * 3]), x_test.reshape([12630, 32 * 32 * 3])

y_train, y_test = to_categorical(y_train), to_categorical(y_test)

model = Sequential()
model.add(Dense(input_dim=32 * 32 * 3, units=500, activation="relu", kernel_regularizer=regularizers.l2(0.0001)))
model.add(Dense(units=500, activation="relu", kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(units=43, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=100, epochs=20)

model.save('my_model.h5')
# %%
score = model.evaluate(x_test, y_test)
print("Total loss on Testing Set: ", score[0])
print("Accuracy of Testing set: ", score[1])
