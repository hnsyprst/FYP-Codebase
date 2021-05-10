
import json
import numpy as np
from scipy import interp
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import tensorflow.keras as keras

DATAPATH = "C:/Users/jrhen/Desktop/data.json"

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
    axs[0].plot(history.history["val_accuracy"], label = "Testing Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_title("Accuracy Evaluation")

    axs[1].plot(history.history["loss"], label = "Training Loss")
    axs[1].plot(history.history["val_loss"], label = "Testing Loss")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].set_title("Loss (Error) Evaluation")

    plt.show()
    
def plotHistoryAlternate(history):
    lw = 2

    plt.plot(history.history["accuracy"], label = "Training Accuracy", lw=lw)
    plt.plot(history.history["val_accuracy"], label = "Validation Accuracy", lw=lw)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(loc="upper right")
    plt.title("Accuracy Evaluation")

    plt.show()

def prepareDatasets(testSize, validationSize):
    # x is inputs (i.e. MFCCs)
    # y is targets (i.e. Labels)
    x, y, mapping = loadData(DATAPATH)

    # create training / testing split (this will prevent cheating - the model appearing better than it is
    # because it is validated on the same data it is evaluated on)
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

if __name__ == "__main__":
    print("I am a classifier with overfitting measures")
    
    # create training, validation and testing sets
    xTrain, xValidation, xTest, yTrain, yValidation, yTest, mapping = prepareDatasets(0.25, 0.2)

    #model = keras.Sequential([
        # input
        # ignores 0th dimension, numSamples
    #    keras.layers.Flatten(input_shape=(xTrain.shape[1], xTrain.shape[2])),

        # hidden layer 1
    #    keras.layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
    #    keras.layers.Dropout(0.3),
        
        # hidden layer 2
    #    keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
    #    keras.layers.Dropout(0.3),
        
        # hidden layer 3
    #    keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
    #    keras.layers.Dropout(0.3),

        # output
    #    keras.layers.Dense(19, activation="softmax")
    #    ])

    model = keras.Sequential([
        # input
        # ignores 0th dimension, numSamples
        keras.layers.Flatten(input_shape=(xTrain.shape[1], xTrain.shape[2])),

        # hidden layer 1
        keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.1)),
        keras.layers.Dropout(0.3),
                
        # hidden layer 2
        keras.layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.1)),
        keras.layers.Dropout(0.3),
        
        # hidden layer 3
        keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.1)),
        keras.layers.Dropout(0.3),

        # output
        keras.layers.Dense(18, activation="softmax")
        ])

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer = optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary()

    history = model.fit(xTrain, yTrain,
                        validation_data=(xValidation, yValidation),
                        epochs=100,
                        batch_size=32)
    
    testLoss, testAccuracy = model.evaluate(xTest, yTest, verbose = 1)
    print("Testing Accuracy: {}".format(testAccuracy))

    targetsTest = label_binarize(yTest, classes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
    
    plotHistoryAlternate(history)
    # plotRoc(predictions, targetsTest, 18, mapping)