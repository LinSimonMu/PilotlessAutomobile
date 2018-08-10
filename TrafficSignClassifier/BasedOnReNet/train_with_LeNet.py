from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np
import matplotlib.pyplot as plt
import pickle
import LeNet_model

# 定义相关参数
NB_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1
OPTIMIZER = Adam()
IMG_ROWS, IMG_COLS = 32, 32
NB_CLASSES = 43
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 3)

# 载入数据
with open("./Data/train.p", mode="rb") as f:
    train = pickle.load(f)
with open("./Data/train.p", mode="rb") as f:
    test = pickle.load(f)

x_train, y_train = train["features"] / 255, train["labels"]
x_test, y_test = test["features"] / 255, test["labels"]

# label one-hot向量化
y_train, y_test = np_utils.to_categorical(
    y_train, NB_CLASSES), np_utils.to_categorical(y_test, NB_CLASSES)

model = LeNet_model.LeNet.bulid(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
model.compile(loss="categorical_crossentropy",
              optimizer=OPTIMIZER, metrics=["accuracy"])
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE,
                    epochs=NB_EPOCH, verbose=VERBOSE)
score = model.evaluate(x_test, y_test, verbose=VERBOSE)
print("Total loss on Testing Set: ", score[0])
print("Accuracy of Testing set: ", score[1])

# 保存模型
model.save('my_model.h5')

# summarize history for accuracy
print(history.history.keys())
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# 保存模型
model.save('my_model.h5')
