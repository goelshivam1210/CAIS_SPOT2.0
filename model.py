from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras import backend as K

import math
import tensorflow as tf
import numpy as np
import cv2
import glob
#from imutils import paths

class ClusterNet:

    # list of annotation file names
    annotations = glob.glob("data/Annotations")
    mom = 0.99
    eps = 0.001

    @staticmethod
    def build():
        
        model = Sequential()
        mom = 0.99
        eps = 0.001
        inputShape = Input(shape = (None, None, 5))
        
        # layer 1
        model.add(Conv2D(20, (3, 3), padding = "same", input_shape = inputShape,
        kernel_initializer = 'random_normal'))
        # activation layer
        model.add(PReLU(alpha_initializer = "zeros"))
        model.add(BatchNormalization(axis = -1, momentum = mom, epsilon = eps ))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = 2))
        
        # layer 2
        model.add(Conv2D(20, (3, 3), padding = "same"))
        model.add(PReLU(alpha_initializer = "zeros"))

        # layer3
        model.add(Conv2D(20, (3, 3), padding = "same"))
        model.add(PReLU(alpha_initializer = "zeros"))
        model.add(BatchNormalization(axis = -1, momentum = mom, epsilon = eps ))

        # layer 4
        model.add(Conv2D(20, (1, 1), padding = "same", kernel_initializer = 'random_normal'))
        model.add(PReLU(alpha_initializer = "zeros"))
        model.add(BatchNormalization(axis = -1, momentum = mom, epsilon = eps ))

        # layer 5
        model.add(Conv2D(20, (3, 3), padding = "same", kernel_initializer = 'random_normal'))
        model.add(PReLU(alpha_initializer = "zeros"))
        model.add(BatchNormalization(axis = -1, momentum = mom, epsilon = eps ))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = 2))

        # layer 6
        model.add(Conv2D(20, (3, 3), padding = "same", kernel_initializer = 'random_normal'))
        model.add(PReLU(alpha_initializer = "zeros"))
        model.add(BatchNormalization(axis = -1, momentum = mom, epsilon = eps ))

        # layer 7
        model.add(Conv2D(20, (1, 1), padding = "same", kernel_initializer = 'random_normal'))
        model.add(PReLU(alpha_initializer = "zeros"))
        model.add(BatchNormalization(axis = -1, momentum = mom, epsilon = eps ))

        return model

'''    
def loss_function(self, y_true, y_pred):
        # calculate the euclidean loss
        # calculate the distance from the feature maps boxes to the x, y ground truth coordinates
        # generate the feature map from the ground truth labels
        y_true  = self.generate_heatmap(y_true)
        # it clips the values for some reason, have to investigate more on this
        y_pred = K.clip(y_pred, K.epsilon(), 1.0-K.epsilon())
        # take the euclidean loss and return it
        return K.sqrt(K.sum(K.square(y_pred - y_true), axis = -1))

    def generate_heatmap(self, frame):
        
        image = np.zeros((self.inputShape[0], self.inputShape[1]))
        # basically populate the matrix with 1's according to the ground truth
        # apply gaussian blur to the image 
        # that should be the heatmap
        for i in range (2, len(annotations), 5):
            annotation = np.load(os.path.join("/data/Images/Annotations/", annotations[i]))
            # need to come up with a way to apply broadcasting or use an np library as this is extremely inefficient
            for notation in annotation:
                for i in range (notation[0], notation[2]):
                    for j in range (notation[1], notation[3]):
                        image [i][j] = 1
            # TODO Work on the kernel size of the gaussian blur and other variables, like sigma and d(downsampling)
            image = cv2.GaussianBlur(image, (5, 5), 0)

        return image
'''

if __name__ == "__main__":
    cn = ClusterNet()

