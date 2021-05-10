
import json
import numpy as np
from scipy import interp
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import tensorflow.keras as keras

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
    plt.rcParams.update({'font.size': 22})

    lw = 4

    plt.plot(history.history["loss"], label = "Training Loss", lw=lw)
    plt.plot(history.history["val_loss"], label = "Testing Loss", lw=lw)
    plt.ylabel("Error")
    plt.xlabel("Epoch")
    plt.title("Loss (Error) Evaluation")

    plt.show()

if __name__ == "__main__":
    print("I am a classifier with overfitting measures")

    inputs, targets, mapping = loadData("C:/Users/jrhen/Desktop/data.json")


    inputsTrain, inputsTest, targetsTrain, targetsTest = train_test_split(inputs,
                                                                          targets,
                                                                          test_size = 0.1)

    model = keras.Sequential([
        # input
        # ignores 0th dimension, numSamples
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),

        # hidden layer 1
        keras.layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        
        # hidden layer 2
        keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        
        # hidden layer 3
        keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # output
        keras.layers.Dense(19, activation="softmax")
        ])

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer = optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary()

    history = model.fit(inputsTrain, targetsTrain,
                        validation_data=(inputsTest, targetsTest),
                        epochs=100,
                        batch_size=32)

    predictions = model.predict(inputsTest, batch_size = 32, verbose = 1)

    targetsTest = label_binarize(targetsTest, classes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
    
    plotHistoryAlternate(history)
    # plotRoc(predictions, targetsTest, 18, mapping)