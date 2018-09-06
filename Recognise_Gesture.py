import cv2, pickle
import numpy as np
import tensorflow as tf
import sqlite3
from keras.models import load_model
import os
import collections
from collections import Counter 


def get_image_size():
	img = cv2.imread('gestures/0/100.jpg', 0)
	return img.shape

def keras_process_image(img):
	img = cv2.resize(img, (image_x, image_y))
	img = np.array(img, dtype=np.float32)
	img = np.reshape(img, (1, image_x, image_y, 1))
	return img

def keras_predict(model, image):
	processed = keras_process_image(image)
	pred_probab = model.predict(processed)[0]
	pred_class = list(pred_probab).index(max(pred_probab))
	return max(pred_probab), pred_class

def get_pred_text_from_db(pred_class):
	conn = sqlite3.connect("gesture_db.db")
	cmd = "SELECT g_name FROM gesture WHERE g_id="+str(pred_class)
	cursor = conn.execute(cmd)
	for row in cursor:
		return row[0]

def get_hand_hist():
	with open("pickle/hist.pickle", "rb") as f:
		hist = pickle.load(f)
	return hist
image_x, image_y = get_image_size()



