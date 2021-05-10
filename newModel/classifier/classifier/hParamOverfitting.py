# HC MEANS HARDCODED - CHANGE BACK TO NON HC TO RETURN TO HYPERBAND TESTING

import json
import io
import numpy as np
from scipy import interp
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.preprocessing import label_binarize
import tensorflow as tf
import tensorflow.keras as keras
from tensorboard.plugins.hparams import api as hp
import datetime
import kerastuner as kt
from kerastuner_tensorboard_logger import (
    TensorBoardLogger,
    setup_tb  # Optional
)

physicalDevices = tf.config.experimental.list_physical_devices('GPU')
assert len(physicalDevices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physicalDevices[0], True)

#DATAPATH = r"D:\Work\UNI\!-FinalYear\FYP\LimitedDataset\data.json"
#NUMGENRES = 2

# Minor HParams
DATAPATH = r"C:\Users\jrhen\Desktop\data.json"
NUMGENRES = 18
DATETIME = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOGDIR = "logs/fit/" + DATETIME
BATCHSIZE = 32
FIGSIZE = (16, 8)
EPOCHS = 100

# HParams
#HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32, 64]), display_name='Neurons in Dense Layer')
#HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.6), display_name='Dropout Probability')
#HP_KERNELS = hp.HParam('kernels', hp.Discrete([16, 32, 64]), display_name='Number of Kernels')
#HP_POOLSIZE = hp.HParam('pool_size', hp.Discrete([1, 2, 3]), display_name='Pool Size (value x value)')
#HP_FILTERSIZE = hp.HParam('filter_size', hp.Discrete([3, 5, 7]), display_name='Filter Size (value x value)')

METRIC_ACCURACY = 'accuracy'


def loadData(datasetPath):
    with open(datasetPath, "r") as path:
        data = json.load(path)

    inputs = np.array(data["MFCC"])
    targets = np.array(data["Labels"])
    mapping = np.array(data["Mapping"])

    return inputs, targets, mapping

def plotRoc(predictions, targetsTest, numClasses, mapping):
    # Plot linewidth.
    figure = plt.figure(figsize = FIGSIZE)

    lw = 2

    # Compute ROC curve and ROC area for each class
    falsePositiveRate = dict()
    truePositiveRate = dict()
    roc_auc = dict()

    for i in range(numClasses):
        falsePositiveRate[i], truePositiveRate[i], _ = sklearn.metrics.roc_curve(targetsTest[:, i], predictions[:, i])
        roc_auc[i] = sklearn.metrics.auc(falsePositiveRate[i], truePositiveRate[i])

    colors = itertools.cycle(['#696969', '#7f0000', '#008000', '#8b008b', '#ff4500', '#ffa500', '#ffff00', '#0000cd', '#00ff00',
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
    
    return figure

def plotTestingAccuracyRecord(testAccuracy):
    plt.text(.0,1.,"{}".format(testAccuracy), bbox={'facecolor':'w','pad':5}, ha="left", va="top", transform=plt.gca().transAxes)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plotHistory(history):
    fig, axs = plt.subplots(2, figsize=FIGSIZE)

    axs[0].plot(history.history["accuracy"], label = "Training Accuracy")
    axs[0].plot(history.history["val_accuracy"], label = "Validation Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_title("Accuracy Evaluation")

    axs[1].plot(history.history["loss"], label = "Training Loss")
    axs[1].plot(history.history["val_loss"], label = "Validation Loss")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].set_title("Loss (Error) Evaluation")

    return fig
    
def plotHistoryAlternate(history):
    lw = 2

    plt.plot(history.history["accuracy"], label = "Training Accuracy", lw=lw)
    plt.plot(history.history["val_accuracy"], label = "Validation Accuracy", lw=lw)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(loc="upper right")
    plt.title("Accuracy Evaluation")

    plt.show()   

def plotConfusionMatrix(cm, mapping):
  figure = plt.figure(figsize = (16, 16))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion Matrix")
  plt.colorbar()
  tickMarks = np.arange(len(mapping))
  plt.xticks(tickMarks, mapping, rotation=45)
  plt.yticks(tickMarks, mapping)

  # Compute the labels from the normalized confusion matrix.
  labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return figure
    
def plot2Image(figure):
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image

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

def buildModel(hParams):
    # get the input shape (ignoring the 0th dimension, numSamples)
    inputShape = (xTrain.shape[1], xTrain.shape[2], xTrain.shape[3])

    # model creation
    model = keras.Sequential()

    # hParams initialisation
    hp_neurons = hParams.Choice(          'neurons',      values=[0,1,2])
    hp_dropout = hParams.Float(             'dropout',      min_value = 0.3,    max_value = 0.7,    step = 0.2)
    hp_regLambda = hParams.Choice(          'regLambda',    values=[1e-1, 1e-2, 1e-3])

    neuron_list = [[512,256,64], [256,128,64], [2048,1024,512]]

    # input
    model.add(keras.layers.Flatten(input_shape=(xTrain.shape[1], xTrain.shape[2])))

    # hidden layer 1
    model.add(keras.layers.Dense(neuron_list[hp_neurons][0], activation="relu", kernel_regularizer=keras.regularizers.l2(hp_regLambda)))
    model.add(keras.layers.Dropout(hp_dropout))
                
    # hidden layer 2
    model.add(keras.layers.Dense(neuron_list[hp_neurons][1], activation="relu", kernel_regularizer=keras.regularizers.l2(hp_regLambda)))
    model.add(keras.layers.Dropout(hp_dropout))
        
    # hidden layer 3
    model.add(keras.layers.Dense(neuron_list[hp_neurons][2], activation="relu", kernel_regularizer=keras.regularizers.l2(hp_regLambda)))
    model.add(keras.layers.Dropout(hp_dropout))

    # output
    model.add(keras.layers.Dense(18, activation="softmax"))

    # compile the model
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer = optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary()
        
    return model

def buildModelHC():
    # get the input shape (ignoring the 0th dimension, numSamples)
    inputShape = (xTrain.shape[1], xTrain.shape[2], xTrain.shape[3])

    # model creation
    model = keras.Sequential()

    # hParams initialisation
    hp_neurons = 992
    hp_dropout = 0.2
    hp_kernels = 16
    hp_poolSize = 1
    hp_filterSize = 9
    hp_learningRate = 0.001
    hp_regLambda = 0.01

    # convolutional layer 1
    # 32 filters, 3x3 kernel, ReLU as activation function
    model.add(keras.layers.Conv2D(hp_kernels, (hp_filterSize, hp_filterSize), activation='relu', input_shape=inputShape, padding='same'))
    # pooling layer 1
    # Max pooling, 3x3 grid, 2x2 stride, 0 padding around all the edges
    model.add(keras.layers.MaxPool2D((hp_poolSize, hp_poolSize), strides=(2, 2), padding='same'))
    # batch normalisation 1
    model.add(keras.layers.BatchNormalization())
    # dropout of 30%
    model.add(keras.layers.Dropout(hp_dropout))
    
    # convolutional layer 1
    # 32 filters, 3x3 kernel, ReLU as activation function
    model.add(keras.layers.Conv2D(hp_kernels, (hp_filterSize, hp_filterSize), activation='relu', input_shape=inputShape, padding='same'))
    # pooling layer 1
    # Max pooling, 3x3 grid, 2x2 stride, 0 padding around all the edges
    model.add(keras.layers.MaxPool2D((hp_poolSize, hp_poolSize), strides=(2, 2), padding='same'))
    # batch normalisation 1
    model.add(keras.layers.BatchNormalization())
    # dropout of 30%
    model.add(keras.layers.Dropout(hp_dropout))
    
    # convolutional layer 3
    # 32 filters, 2x2 kernel, ReLU as activation function
    model.add(keras.layers.Conv2D(hp_kernels, (2, 2), activation='relu', input_shape=inputShape, padding='same'))
    # pooling layer 3
    # Max pooling, 2x2 grid, 2x2 stride, 0 padding around all the edges
    model.add(keras.layers.MaxPool2D((hp_poolSize, hp_poolSize), strides=(2, 2), padding='same'))
    # batch normalisation 3
    model.add(keras.layers.BatchNormalization())
    # dropout of 30%
    model.add(keras.layers.Dropout(hp_dropout))

    # flatten the multidimensional convolution data down to a single dimension
    model.add(keras.layers.Flatten())

    # hidden layer 1
    # 64 neurons, ReLU as activation function, L2 regularisation with a lambda of 0.001
    model.add(keras.layers.Dense(hp_neurons, activation='relu', kernel_regularizer=keras.regularizers.l2(hp_regLambda)))
    # dropout of 30%
    model.add(keras.layers.Dropout(hp_dropout))

    # output layer
    # softmax as activation function
    model.add(keras.layers.Dense(NUMGENRES, activation='softmax'))

    # compile the model
    optimizer = keras.optimizers.Adam(learning_rate=hp_learningRate)
    model.compile(optimizer = optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary()
        
    return model


class hParamsCallback(keras.callbacks.Callback):
    def __init__(self, run_dir, hparams, inputShape):
       super().__init__()
       self.run_dir = run_dir
       self.hparams = hparams
    def on_epoch_end(self, epoch, logs=None):
        # Use the model to predict the values from the test dataset.
        rawPredictions = self.model.predict(xTest, batch_size = BATCHSIZE, verbose = 1)
        predictions = np.argmax(rawPredictions, axis=1)

        # Calculate the confusion matrix.
        cm = sklearn.metrics.confusion_matrix(yTest, predictions, normalize='true')
        # Log the confusion matrix as an image summary.
        cmFigure = plotConfusionMatrix(cm, mapping)
        cmTensorFigure = plot2Image(cmFigure)
    
        binyTest = label_binarize(yTest, classes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
        #binyTest = label_binarize(yTest, classes=[0,1,2])

        rocFigure = plotRoc(rawPredictions, binyTest, NUMGENRES, mapping=mapping)
        rocTensorFigure = plot2Image(rocFigure)

        # Log the confusion matrix as an image summary.
        with self.file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cmTensorFigure, step=epoch)
        with self.file_writer_roc.as_default():
            tf.summary.image("ROC AUC", rocTensorFigure, step=epoch)

def run(run_dir, hparams, inputShape):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = buildModel(inputShape, hparams, run_dir)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

if __name__ == "__main__":
    print("I am the hyperparameter tuning classifier using the Keras tuner - overfitting")

    # create training, validation and testing sets
    xTrain, xValidation, xTest, yTrain, yValidation, yTest, mapping = prepareDatasets(0.25, 0.2)
    
    # create callbacks
    tensorboardCallback = tf.keras.callbacks.TensorBoard(log_dir=LOGDIR,
                                                         histogram_freq=1,
                                                         embeddings_freq=1,
                                                         write_graph=True,
                                                         update_freq='batch')
    earlyStopCallback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        patience=5)
    # instantiate the tuner
    tuner = kt.Hyperband(buildModel,
                         objective='val_accuracy',
                         max_epochs=100,
                         factor=3,
                         directory=LOGDIR,
                         project_name='tuning',
                         logger=TensorBoardLogger(metrics=['val_accuracy'], logdir=LOGDIR + '/hparams'))
    setup_tb(tuner)
    # run the tuner
    tuner.search(xTrain, yTrain,
                 validation_data=(xValidation, yValidation),
                 epochs=EPOCHS,
                 batch_size= BATCHSIZE,
                 callbacks=[earlyStopCallback])
    bestHParams = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""The hyperparameter search is complete.""")

    # build the optimal hypermodel
    model = tuner.hypermodel.build(bestHParams)

    #model = buildModelHC()

    # train the hypermodel on the training and validation sets
    # X axis is inputs (i.e. MFCCs)
    # Y axis is targets (i.e. Labels)
    history = model.fit(xTrain, yTrain,
                        validation_data=(xValidation, yValidation),
                        epochs=EPOCHS,
                        batch_size= BATCHSIZE,
                        callbacks=[tensorboardCallback, earlyStopCallback])
    
    # evaluate the hypermodel 
    testLoss, testAccuracy = model.evaluate(xTest, yTest, verbose = 1)
    print("Testing Accuracy: {}".format(testAccuracy))
    
    # log the evaulation in TensorBoard
    file_writer_ra = tf.summary.create_file_writer(LOGDIR + '/ra')
    with file_writer_ra.as_default():
        tf.summary.text('Real Accuracy', str(testAccuracy), step=0, description='Accuracy on unseen data')

    rawPredictions = model.predict(xTest, batch_size = BATCHSIZE, verbose = 1)
    predictions = np.argmax(rawPredictions, axis=1)

    # plot the model history
    historyPlot = plotHistory(history)
    file_writer_history = tf.summary.create_file_writer(LOGDIR + '/history')
    # log the history as an image summary
    with file_writer_history.as_default():
        tf.summary.image("Model History", plot2Image(historyPlot), step=0)

    # plot the confusion matrix
    # calculate the confusion matrix
    cm = sklearn.metrics.confusion_matrix(yTest, predictions, normalize='true')
    confusionPlot = plotConfusionMatrix(cm, mapping)
    # log the confusion matrix as an image summary
    file_writer_cm = tf.summary.create_file_writer(LOGDIR + '/cm')
    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", plot2Image(confusionPlot), step=0)
    
    # plot the roc auc
    # reshape (binarise) the dataset labels
    binyTest = label_binarize(yTest, classes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
    #binyTest = label_binarize(yTest, classes=[0,1,2])

    rocPlot = plotRoc(rawPredictions, binyTest, NUMGENRES, mapping=mapping)
    file_writer_roc = tf.summary.create_file_writer(LOGDIR + '/roc')
    with file_writer_roc.as_default():
        tf.summary.image("ROC AUC", plot2Image(rocPlot), step=0)
        
    model.save(LOGDIR + '/save')