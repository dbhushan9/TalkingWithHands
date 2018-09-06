from sklearn.metrics import classification_report,confusion_matrix
import sqlite3
import os
import tensorflow as tf 
import numpy as np
import pickle
from keras.models import load_model
from keras.utils import np_utils
import matplotlib.pyplot as plt
import itertools


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

def get_pred_text_from_db(pred_class):
  conn = sqlite3.connect("gesture_db.db")
  cmd = "SELECT g_name FROM gesture WHERE g_id="+str(pred_class)
  cursor = conn.execute(cmd)
  for row in cursor:
    return row[0]


with open("pickle/train_images.pickle", "rb") as f:
  train_images = np.array(pickle.load(f))
with open("pickle/train_labels.pickle", "rb") as f:
  train_labels = np.array(pickle.load(f), dtype=np.int32)

with open("pickle/test_images.pickle", "rb") as f:
  test_images = np.array(pickle.load(f))
with open("pickle/test_labels.pickle", "rb") as f:
  test_labels = np.array(pickle.load(f), dtype=np.int32)


train_images = np.reshape(train_images, (train_images.shape[0], 50,50, 1))
test_images = np.reshape(test_images, (test_images.shape[0], 50, 50, 1))
train_labels = np_utils.to_categorical(train_labels)
test_labels = np_utils.to_categorical(test_labels)



model = load_model('cnn_model_keras2.h5')

pred_class = model.predict(test_images)
#print(pred_class)
pred_class = np.argmax(pred_class, axis=1)
#print(pred_class)
   
target_names = []
conn = sqlite3.connect("gesture_db.db")
cmd = "SELECT g_name FROM gesture"
cursor = conn.execute(cmd)
for row in cursor:
  target_names.append(row[0])
#print(target_names);


print(classification_report(np.argmax(test_labels,axis=1), pred_class,target_names=target_names))

print(confusion_matrix(np.argmax(test_labels,axis=1), pred_class))


# Plotting the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.savefig("confusion-matrix3.jpg")

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = (confusion_matrix(np.argmax(test_labels,axis=1), pred_class))

np.set_printoptions(precision=2)

plt.savefig("confusion-matrix1.jpg")
plt.figure()


# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix')
#plt.figure()
# Plot normalized confusion matrix
#plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
#                      title='Normalized confusion matrix')
#plt.figure()
plt.savefig("confusion-matrix2.jpg")
plt.show()
'''
'''