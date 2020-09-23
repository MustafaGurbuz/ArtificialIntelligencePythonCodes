"""
#SORU 3
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv("F:/YAZILIM MÜHENDİSLİĞİ/4.SINIF/Yapay Zeka/LAB/Ödevler/Python Dosyaları/yaprak_veriseti/train.csv")
test = pd.read_csv("F:/YAZILIM MÜHENDİSLİĞİ/4.SINIF/Yapay Zeka/LAB/Ödevler/Python Dosyaları/yaprak_veriseti/test.csv")

label_encoder = LabelEncoder().fit(train.species)
labels = label_encoder.transform(train.species)
classes = list(label_encoder.classes_)

train = train.drop(["id","species"],axis=1)
test = test.drop(["id"],axis=1)
nb_features = 192
nb_classes = len(classes)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler().fit(train.values)
train = sc.transform(train.values)

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(train,labels, test_size=0.1)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)

X_train = np.array(X_train).reshape(891,192,1)
X_valid = np.array(X_valid).reshape(99,192,1)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv1D, Dropout, MaxPooling1D, Flatten

#1.SORU CEVABI
model = Sequential()
model.add(Conv1D(1024,1,input_shape=(nb_features,1)))
model.add(Activation("relu"))
model.add(MaxPooling1D(2))
model.add(Conv1D(512,1))
model.add(MaxPooling1D(1))
model.add(Conv1D(256,1))
model.add(MaxPooling1D(2))
model.add(Conv1D(128,1))
model.add(MaxPooling1D(2))
model.add(Dropout(0.35))
model.add(Flatten())
model.add(Dense(1000,activation="relu"))
model.add(Dense(1500,activation="relu"))
model.add(Dense(2000,activation="relu"))
model.add(Dense(nb_classes, activation="softmax"))
model.summary()

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
model.fit(X_train,y_train,validation_data=(X_valid,y_valid),epochs=25)

print("Ortalama Eğitim kaybı: ",np.mean(model.history.history["loss"]))
print("Ortalama Eğitim başarımı: ",np.mean(model.history.history["acc"]))
print("Ortalama Doğrulama kaybı: ",np.mean(model.history.history["val_loss"]))
print("Ortalama Doğrulama başarımı: ",np.mean(model.history.history["val_acc"]))

import matplotlib.pyplot as pt

fig, (ax1, ax2) = pt.subplots(2, 1, figsize=(15, 15))
ax1.plot(model.history.history['loss'], color='g', label="Eğitim Kaybı")
ax1.plot(model.history.history['val_loss'], color='y', label="Doğrulama Kaybı")
ax1.set_xticks(np.arange(20,100,20))
ax2.plot(model.history.history['acc'], color='b', label="Eğitim Başarımı")
ax2.plot(model.history.history['val_acc'], color='r', label="Doğrulama Başarımı")
ax2.set_xticks(np.arange(20,100,20))
pt.legend()
pt.show()
"""