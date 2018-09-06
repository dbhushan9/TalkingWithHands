import numpy as np
import pickle
import cv2, os
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import backend as K
K.set_image_dim_ordering('tf')
import matplotlib.pyplot as plt
from keras.utils import np_utils


def get_featuremaps(model, layer_idx, X_batch):
	get_activations = K.function([model.layers[0].input,K.learning_phase()],[model.layers[layer_idx].output,])
	activations = get_activations([X_batch,0])
	return activations

def get_image_size():
	img = cv2.imread('gestures/0/100.jpg', 0)
	#print("dasdasfasf",img.shape)
	return img.shape

image_x, image_y = get_image_size()

def keras_process_image(img):
	img = cv2.resize(img, (image_x, image_y))
	img = np.array(img, dtype=np.float32)
	#print(img.shape)
	img = np.reshape(img, (1, image_x, image_y, 1))
	return img


layer_num=16
filter_num=0

PATH = os.getcwd()

test_image = cv2.imread('gestures/0/100.jpg', 0)

print("Shape",test_image.shape)
model = load_model('cnn_model_keras2.h5')
feature_maps = []
fig=plt.figure(figsize=(16,16))	
plt.title("featuremaps-layer-{}".format("vgg-16"))
for layer_num in range(16):
	activations = get_featuremaps(model, int(layer_num),keras_process_image(test_image))
	feature_maps.append( activations[0][0] )     
	num_of_featuremaps= 10  #feature_maps.shape[2]
	
	
subplot_num=int(np.ceil(np.sqrt(num_of_featuremaps)))
j=-1
for i in range(int(num_of_featuremaps * 16)):
	ax = fig.add_subplot(16, num_of_featuremaps, i+1)
	#ax.imshow(output_image[0,:,:,i],interpolation='nearest' ) #to see the first filter
	if i%10 == 0:
		j = j+1
	ax.imshow(feature_maps[j][:,:,i%10],cmap='gray')
	plt.xticks([])
	plt.yticks([])
	plt.tight_layout()



print("now showing")
plt.savefig("featuremaps-layer-{}".format(layer_num) + '.jpg')
plt.show()
print("now saving")
print("done saving")
