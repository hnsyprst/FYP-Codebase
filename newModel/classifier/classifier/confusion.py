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
    print("Hello")
    plt.show()
    return figure

mapping = ["Baile + Dancehall", "Bass + Breaks", "Club + Dubstep", "Disco", "Drum and Bass + Jungle", "Electro + Techno", "Footwork", "Garage + House", "Grime + Hip Hop", "UK Funky", "UK Hardcore"]

cm = np.array([ [0.46,0.03,0.10,0.01,0.08,0.09,0.03,0.04,0.10,0.06,0.00],
                [0.00,0.28,0.11,0.00,0.15,0.20,0.00,0.19,0.04,0.01,0.02],
                [0.02,0.16,0.20,0.00,0.07,0.25,0.02,0.17,0.04,0.04,0.02],
                [0.05,0.00,0.05,0.83,0.00,0.00,0.00,0.00,0.02,0.05,0.00],
                [0.00,0.08,0.06,0.00,0.61,0.15,0.02,0.02,0.00,0.01,0.04],
                [0.00,0.14,0.04,0.00,0.06,0.44,0.03,0.18,0.03,0.06,0.03],
                [0.02,0.14,0.10,0.00,0.10,0.12,0.33,0.10,0.02,0.02,0.05],
                [0.05,0.03,0.11,0.07,0.04,0.17,0.02,0.43,0.01,0.05,0.01],
                [0.01,0.09,0.14,0.01,0.02,0.07,0.02,0.01,0.53,0.09,0.01],
                [0.10,0.00,0.17,0.02,0.05,0.00,0.02,0.36,0.02,0.26,0.00],
                [0.00,0.06,0.11,0.00,0.04,0.40,0.00,0.06,0.00,0.02,0.30],
            ])

confusionPlot = plotConfusionMatrix(cm, mapping)