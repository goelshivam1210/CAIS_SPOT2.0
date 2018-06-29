import matplotlib
matplotlib.use("Agg")

from keras import optimizers
from keras import losses
from keras.preprocessing.image import img_to_array
from keras.utils import multi_gpu_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasRegressor
import matplotlib.pyplot as plt
#from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os
import model
import keras
#from pathlibs import paths

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model",
	help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output accuracy/loss plot")
ap.add_argument("-n", "--frames", type = int, required = True, default = 5,
        help = "Number of frames in the input stack")
ap.add_argument("-g", "--gpu", type = int, default = 1,
        help = "Number of GPUs to parallelize the process")
ap.add_argument("-a", "--annotations", help = "path to the annotation file")
args = vars(ap.parse_args())



# this function generates the heatmap of the frame
# takes argument image dimensions and name of the frame to pick annotations from
def generateHeatmap(image_dims, name):
    # creates a numpy array of zeros of the same dimensions as image
    image = np.zeros((image_dims))
    name = name[:-4] 
    name = name + ".npy"
    print (name)
    #print (os.path.join(args["annotations"], name))
    annotation = np.load(os.path.join(args["annotations"], name), encoding = 'ASCII')
    print (annotation)
    # for every annotation in the file we change the values to 1
    for notation in annotation:
        #print (int(notation[0]
        print (notation[0], notation[1], notation[2], notation[3])
        for i in range (int(notation[0]), int(notation[2])):
            for j in range (int(notation[1]), int(notation[3])):
                image[i][j] = 1
    # apply Gaussian Blur to the image            
    image = cv2.GaussianBlur(image, (5, 5), 0)

    return image



EPOCHS = 100
LR = 0.01
BS = 8
#IMAGE_DIMS = (576, 720)

data = []
Y = []

list_of_dir = os.listdir(args["dataset"])
#print (list_of_dir)
list_of_paths = []
for dirs in list_of_dir:
    list_of_paths.append(os.path.join(args["dataset"], dirs))
#print (list_of_paths)
    
# labels = []
for path in list_of_paths:
    images_list = os.listdir(path)
    #print (images_list)
    # make sure the size of array is consistent 
    if len(images_list) % 5 != 0:
        diff = len(images_list) % 5
        images_list = images_list[:len(images_list)-diff]
    #print (images_list)
    for im in range(0, len(images_list), 5):
        #print (im)
        dims = np.shape(cv2.imread(os.path.join(path, images_list[im])))
        
        datastack = np.empty((dims))
        #print (datastack)
        for i in range(im, im+args["frames"]):
            #print (i)
            #print (im)
                
            path_to_image = os.path.join(path, images_list[im])
            image = cv2.imread(path_to_image)
            #print (path_to_image)
            image = img_to_array(image)
            # print (path_to_image)
            # removes the third dimension of the image 
            image = image[:, :, 0]
            # to generate heatmap for every central frame
            if i == 2:
                print (images_list[i])
                print (image.shape)
                y_i = generateHeatmap(image.shape, images_list[im+i])
            #image = img_to_array(image)
            #print(image.shape)
            datastack = np.dstack((datastack, image))
        #print (datastack.shape)
        # this updates the list of ground truths
        Y.append(y_i)
        datastack = datastack[:, :, 3:]
        '''
        for i in range(0, 5):
            print(datastack[0][0][i])
        print ("")
        #import sys
        #sys.exit(1)
        '''
        data.append(datastack)
                
        #print (datastack.shape)
print (data.shape)    

print ("[INFO] Compiling Model...")

model_cluster = model.ClusterNet.build()
opt = keras.optimizers.SGD(lr = 0.01, decay = 0.1, nesterov = True)
#parallel_model = multi_gpu_model(model_cluster, gpus = args["gpu"])
model_cluster.compile(loss = 'mean_squared_error', optimizer = opt)

# train the network
print ("[INFO] Training the Network...")
H = model.fit_generator((data), y = Y,  batch_size = BS,
    steps_per_epoch = len(data) // BS,   
    epochs = EPOCHS, verbose = 1)

print ("[INFO] serializing network...")
model.save(args["model"])

# plotting

plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arrange(0, N), H.history["loss"], label = "train_loss")
plt.title("Training loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc = "upperleft")
plt.savefig(args["plot"])

'''
# takes argument image dimensions and name of the frame to pick annotations from
def generateHeatmap(image_dims, name):
    # creates a numpy array of zeros of the same dimensions as image
    image = np.zeros((image_dims))
    annotation = np.load(os.path.join(args["annotation"], name))
    # for every annotation in the file we change the values to 1
    for notation in annotation:
        for i in range (notation[0], notation[2]):
            for j in range (notation[1], notation[3]):
                image[i][j] = 1
    # apply Gaussian Blur to the image            
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    return image
   ''' 
    

