from matplotlib import pyplot
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
import time


def load_dataset():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
    print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))
    return x_train, y_train, x_test, y_test


def prep_pixels(train, test):
    train_norm = train.astype('float32') / 255.0
    test_norm = test.astype('float32') / 255.0
    return train_norm, test_norm


def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform',
                     padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def define_model2():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform',
                     padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform',
                     padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu',
                     kernel_initializer='he_uniform',
                     padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu',
                     kernel_initializer='he_uniform',
                     padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu',
                    kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def summarize_diagnostics(history):
    pyplot.subplot(211)
    pyplot.title("Cross Entropy Loss")
    pyplot.plot(history.history["loss"], color="blue", label="train")
    pyplot.plot(history.history["val_loss"], color="orange", label="test")

    pyplot.subplot(212)
    pyplot.title("Classification Accuracy")
    pyplot.plot(history.history["accuracy"], color="blue", label="train")
    pyplot.plot(history.history["val_accuracy"], color="orange", label="test")

    pyplot.show()
    pyplot.close()


def run_test_harness():
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print("Start time: " + current_time)
    x_train, y_train, x_test, y_test = load_dataset()

    x_train, x_test = prep_pixels(x_train, x_test)

    model = define_model2()

    history = model.fit(x_train, y_train, epochs=100, batch_size=64,
                        validation_data=(x_test, y_test), verbose="0")

    _, acc = model.evaluate(x_test, y_test, verbose="0")
    print('> %.3f' % (acc * 100.0))

    tn = time.localtime()
    current_time = time.strftime("%H:%M:%S", tn)
    print("End time: " + current_time)

    summarize_diagnostics(history)
