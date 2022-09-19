'''
If tensorflow is not installed write in cmd:

pip install tensorflow
pip install --upgrade tensorflow-hub

'''

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
from scipy.spatial import distance
import sys

model_url = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2" # Link to model weights 
IMAGE_SHAPE = (224, 224)

layer = hub.KerasLayer(model_url, input_shape=IMAGE_SHAPE+(3,)) # Loading model weights
model = tf.keras.Sequential([layer]) # Building model object using loaded weights

# img_1 = sys.argv[1]
# img_2 = sys.argv[2]
img_1='screenshot.jpg'
img_2='images/O.jpg'

# Calculations for image 1
img_1 = Image.open(img_1).convert('L').resize(IMAGE_SHAPE) # Resizing the image to required size
img_1 = np.stack((img_1,)*3, axis=-1) # Converting the image into a color representation for each pixel
img_1 = np.array(img_1)/255.0 # Normalizing the values between 0 and 1
embedding_img1 = model.predict(img_1[np.newaxis, ...]) # Extracting the features
embedding_img1_np = np.array(embedding_img1) # Converting to numpy array
flattened_feature_img1 = embedding_img1_np.flatten() # Converting matrix to a vector

# Calculations for image 2
img_2 = Image.open(img_2).convert('L').resize(IMAGE_SHAPE) # Resizing the image to required size
img_2 = np.stack((img_2,)*3, axis=-1) # Converting the image into a color representation for each pixel
img_2 = np.array(img_2)/255.0 # Normalizing the values between 0 and 1
embedding_img2 = model.predict(img_2[np.newaxis, ...]) # Extracting the features
embedding_img2_np = np.array(embedding_img2) # Converting to numpy array
flattened_feature_img2 = embedding_img2_np.flatten() # Converting matrix to a vector

methods = ['sqeuclidean', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine',
'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski',
'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
'sokalmichener', 'sokalsneath', 'braycurtis', 'yule']

mins = [0,0,0,0,0,0,0,0,0,0,0,0.475,0,0,0,0.475,0,0,0,0]
maxes = [150,450, 2.5, 200, .5, .4, .4, 13, .6, 1, 0.5, 1, .6, 13, .4, .8, .5, .7, .5, .3]
scores = []

for i, m in enumerate(methods):

    metric = m # Try using any one of the methods listed above if needed
    dist_boyboy = distance.cdist([flattened_feature_img1], [flattened_feature_img2],
                                    metric)[0]      # Finding similarity between boy-boy

   
    score = max(0, maxes[i] - dist_boyboy[0])
    interval = maxes[i] - mins[i]
    score = score / interval * 100
    scores.append(score)

# print(scores)
print('hehehe',max(scores))
