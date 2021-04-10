# Plant-Desease-Detection-Using-AI
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import cv2
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from PIL import Image
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
path="E:/Coading Part/CSV Files/Cucumber/training/"
xlist=[]
ylist=[]
labels_name={'Ill_cucumber':0,'good_Cucumber':1}
for directory in os.listdir(path):
    for file in os.listdir(path+directory):
        print(path+directory+"\""+file)
        label =labels_name[directory]
        input_img = cv2.imread(path+directory+"/"+file)
        input_img_re =cv2.resize(input_img,(224,224))
        xlist.append(input_img_re)
        ylist.append(label)

# Convolution Neural Networks (CNN)

classifier.add(Conv2D(64,(3,3),input_shape=(224,224,3),activation ="relu",padding='same'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(ZeroPadding2D(padding=(1,1),data_format=None))
classifier.add(Conv2D(64,(3,3),activation ="relu",padding='same'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(ZeroPadding2D(padding=(1,1),data_format=None))
classifier.add(Conv2D(64,(3,3),activation ="relu",padding='same'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(ZeroPadding2D(padding=(1,1),data_format=None))
classifier.add(Conv2D(64,(3,3),activation ="relu",padding='same'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(ZeroPadding2D(padding=(1,1),data_format=None))
classifier.add(Conv2D(64,(3,3),activation ="relu",padding='same'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Conv2D(64,(3,3),activation ="relu",padding='same'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=98, activation='relu'))
classifier.add(Dense(units=88, activation='relu'))
classifier.add(Dense(units=68, activation='relu'))
classifier.add(Dense(units=38, activation='relu'))
classifier.add(Dense(units=2, activation='softmax'))
classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
classifier.save_weights("finall.h5")
hist=classifier.fit(x_train,y_train,batch_size=10,epochs=40,verbose=1,validation_data=(x_test,y_test))
classifier.save("model3.h5")
classifier.load_weights("model3.h5")

# Graph Visualization

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['accuracy']
val_acc=hist.history['val_accuracy']
xc=range(40)
plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel("Num of Epochs")
plt.ylabel("Loss")
plt.title("Train loss vs val loss")
plt.grid(True)
plt.legend(['Train','val'])
plt.style.use(['classic'])

# Results

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.title("Train loss vs val loss")
plt.grid(True)
plt.legend(['Train','val'],loc=4)
plt.style.use(['classic'])

