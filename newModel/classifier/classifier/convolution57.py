import json
import numpy as np
from scipy import interp
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import tensorflow as tf
import tensorflow.keras as keras
import datetime

#DATAPATH = r"D:\Work\UNI\!-FinalYear\FYP\LimitedDataset\data.json"
#NUMGENRES = 2

# Minor HParams
DATAPATH = r"C:\Users\jrhen\Desktop\data.json"
NUMGENRES = 18
LOGDIR = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
BATCHSIZE = 32

# HParams

hDROPOUT = 0.4
hEPOCHS = 100


def loadData(datasetPath):
    with open(datasetPath, "r") as path:
        data = json.load(path)

    inputs = np.array(data["MFCC"])
    targets = np.array(data["Labels"])
    mapping = np.array(data["Mapping"])

    return inputs, targets, mapping

def plotRoc(predictions, targetsTest, numClasses, mapping):
    # Plot linewidth.
    lw = 2

    # Compute ROC curve and ROC area for each class
    falsePositiveRate = dict()
    truePositiveRate = dict()
    roc_auc = dict()

    for i in range(numClasses):
        falsePositiveRate[i], truePositiveRate[i], _ = roc_curve(targetsTest[:, i], predictions[:, i])
        roc_auc[i] = auc(falsePositiveRate[i], truePositiveRate[i])

    colors = cycle(['#696969', '#7f0000', '#008000', '#8b008b', '#ff4500', '#ffa500', '#ffff00', '#0000cd', '#00ff00',
                    '#00fa9a', '#4169e1', '#e9967a', '#00ffff', '#00ffff', '#ff00ff', '#f0e68c', '#dda0dd', '#ff1493'])
    for i, color in zip(range(numClasses), colors):
        plt.plot(falsePositiveRate[i], truePositiveRate[i], color=color, lw=lw,
                 label='{0} (area = {1:0.2f})'
                 ''.format(mapping[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass Receiver Operating Characteristic Graph')
    plt.legend(loc="lower right")
    plt.show()

def plotHistory(history):
    fig, axs = plt.subplots(2)

    axs[0].plot(history.history["accuracy"], label = "Training Accuracy")
    axs[0].plot(history.history["val_accuracy"], label = "Validation Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_title("Accuracy Evaluation")

    axs[1].plot(history.history["loss"], label = "Training Loss")
    axs[1].plot(history.history["val_loss"], label = "Validation Loss")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].set_title("Loss (Error) Evaluation")

    plt.show()
    
def plotHistoryAlternate(history):
    lw = 2

    plt.plot(history.history["loss"], label = "Training Loss", lw=lw)
    plt.plot(history.history["val_loss"], label = "Validation Loss", lw=lw)
    plt.ylabel("Error")
    plt.xlabel("Epoch")
    plt.title("Loss (Error) Evaluation")

    plt.show()

def prepareDatasets(testSize, validationSize):
    # x is inputs (i.e. MFCCs)
    # y is targets (i.e. Labels)
    x, y, mapping = loadData(DATAPATH)

    # create training / testing split (this will prevent cheating - the model appearing better than it is
    # because it is validated on the same data it is evaluated on
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = testSize)

    # create training / validation split
    xTrain, xValidation, yTrain, yValidation = train_test_split(xTrain, yTrain, test_size = validationSize)

    # add a 3rd dimension to the data (tensorflow expects a 3rd dimension for convolutional layers because
    # the input is usually images (which usually have 3 dimensions  - the X, the Y and the channel (red, green or blue)))
    #
    # the data will actually now be 4 dimensional - but the 0th dimension, numSamples, would always have been ignored
    xTrain = xTrain[..., np.newaxis]
    xValidation = xValidation[..., np.newaxis]
    xTest = xTest[..., np.newaxis]

    return xTrain, xValidation, xTest, yTrain, yValidation, yTest, mapping

def buildModel(inputShape):
    # model creation
    model = keras.Sequential()

    # convolutional layer 1
    # 32 filters, 3x3 kernel, ReLU as activation function
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=inputShape, padding='same'))
    # pooling layer 1
    # Max pooling, 3x3 grid, 2x2 stride, 0 padding around all the edges
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    # batch normalisation 1
    model.add(keras.layers.BatchNormalization())
    # dropout of 30%
    model.add(keras.layers.Dropout(0.4))
    
    # convolutional layer 2
    # 32 filters, 3x3 kernel, ReLU as activation function
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=inputShape, padding='same'))
    # pooling layer 2
    # Max pooling, 3x3 grid, 2x2 stride, 0 padding around all the edges
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    # batch normalisation 2
    model.add(keras.layers.BatchNormalization())
    # dropout of 30%
    model.add(keras.layers.Dropout(0.4))
    
    # convolutional layer 3
    # 32 filters, 2x2 kernel, ReLU as activation function
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=inputShape, padding='same'))
    # pooling layer 3
    # Max pooling, 2x2 grid, 2x2 stride, 0 padding around all the edges
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    # batch normalisation 3
    model.add(keras.layers.BatchNormalization())
    # dropout of 30%
    model.add(keras.layers.Dropout(0.4))

    # flatten the multidimensional convolution data down to a single dimension
    model.add(keras.layers.Flatten())

    # hidden layer 1
    # 64 neurons, ReLU as activation function, L2 regularisation with a lambda of 0.001
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    # dropout of 30%
    model.add(keras.layers.Dropout(0.4))

     # hidden layer 1
    # 64 neurons, ReLU as activation function, L2 regularisation with a lambda of 0.001
    model.add(keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    # dropout of 30%
    model.add(keras.layers.Dropout(0.4))

    # output layer
    # softmax as activation function
    model.add(keras.layers.Dense(NUMGENRES, activation='softmax'))

    return model

if __name__ == "__main__":
    print("I am a classifier with convolution (accuracy 0.57)")

    physicalDevices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physicalDevices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physicalDevices[0], True)

    # create training, validation and testing sets
    xTrain, xValidation, xTest, yTrain, yValidation, yTest, mapping = prepareDatasets(0.25, 0.2)

    # get the input shape (ignoring the 0th dimension, numSamples)
    inputShape = (xTrain.shape[1], xTrain.shape[2], xTrain.shape[3])

    # build the model
    model = buildModel(inputShape)

    # compile the model
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer = optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary()

    tensorboardCallback = tf.keras.callbacks.TensorBoard(log_dir=LOGDIR, histogram_freq=1)


    # train the model on the training and validation sets
    # X axis is inputs (i.e. MFCCs)
    # Y axis is targets (i.e. Labels)
    history = model.fit(xTrain, yTrain,
                        validation_data=(xValidation, yValidation),
                        epochs=100,
                        batch_size= BATCHSIZE,
                        callbacks=[tensorboardCallback])
    
    # evaluate the model 
    testLoss, testAccuracy = model.evaluate(xTest, yTest, verbose = 1)
    print("Testing Accuracy: {}".format(testAccuracy))

    # plot results
    predictions = model.predict(xTest, batch_size = BATCHSIZE, verbose = 1)

    yTest = label_binarize(yTest, classes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
    
    plotHistoryAlternate(history)
    plotRoc(predictions, yTest, NUMGENRES, mapping)