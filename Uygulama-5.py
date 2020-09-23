
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv("F:/YAZILIM MÜHENDİSLİĞİ/4.SINIF/Yapay Zeka/LAB/Ödevler/Python Dosyaları/Verisetleri/yaprak_veriseti/train.csv")
test = pd.read_csv("F:/YAZILIM MÜHENDİSLİĞİ/4.SINIF/Yapay Zeka/LAB/Ödevler/Python Dosyaları/Verisetleri/yaprak_veriseti/test.csv")

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
model.add(Conv1D(128,1))
model.add(MaxPooling1D(2))
model.add(Dropout(0.15))
model.add(Flatten())
model.add(Dense(250,activation="relu"))
model.add(Dense(700,activation="relu"))
model.add(Dense(1000,activation="relu"))
model.add(Dense(nb_classes, activation="softmax"))
model.summary()

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
model.fit(X_train,y_train,validation_data=(X_valid,y_valid),epochs=5)

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

#2.SORU CEVABI
from sklearn.model_selection import cross_val_score
from sklearn import svm
clf = svm.SVC(kernel="linear",C=1)
scores=cross_val_score(clf,train,labels,cv=5,scoring="accuracy")
pt.plot(model.history.history["acc"])
pt.title("Model Başarımları")
pt.xlabel("Epok Sayısı")
pt.ylabel("Başarım")


#3.SORU CEVABI
X_train = np.array(X_train).reshape(891,192)
X_valid = np.array(X_valid).reshape(99,192)


from sklearn.linear_model import LogisticRegression

mod = LogisticRegression()

mod.fit(X_train,y_train)
y_pred = mod.predict(X_valid)

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score

a = accuracy_score(y_valid, y_pred)
b = confusion_matrix(y_valid, y_pred)
c = f1_score(y_valid, y_pred)
d = precision_score(y_valid, y_pred)
e = recall_score(y_valid, y_pred)

print(a)
print(b)
print(c)
print(d)
print(e)
"""

"""
#4.SORU CEVABI
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

diyabet = pd.read_csv("F:/YAZILIM MÜHENDİSLİĞİ/4.SINIF/Yapay Zeka/LAB/Ödevler/Python Dosyaları/diyabet.csv")

le = LabelEncoder().fit(diyabet.clas)
labs = le.transform(diyabet.clas)
classes = list(le.classes_)

X=diyabet.drop(["clas"],axis=1)
y=labs


from sklearn.preprocessing import StandardScaler

sa = StandardScaler()
X = sa.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =\
    train_test_split(X, y, test_size=0.33, random_state=0)
    
X_train = np.array(X_train).reshape(514,8,1)
X_test = np.array(X_test).reshape(254,8,1)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


nb_features = 8
nb_classes = len(classes)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv1D, Dropout, MaxPooling1D, Flatten

#1.SORU CEVABI
model = Sequential()
model.add(Conv1D(512,1,input_shape=(nb_features,1)))
model.add(Activation("relu"))
model.add(MaxPooling1D(2))
model.add(Conv1D(384,1))
model.add(MaxPooling1D(2))
model.add(Conv1D(1024,1))
model.add(MaxPooling1D(2))
model.add(Dropout(0.20))
model.add(Flatten())
model.add(Dense(2000,activation="relu"))
model.add(Dense(3000,activation="relu"))
model.add(Dense(nb_classes, activation="softmax"))
model.summary()

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=20)

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

"""
#4.SORU F1-SCORE
import pandas as pd

diyabet = pd.read_csv("F:/YAZILIM MÜHENDİSLİĞİ/4.SINIF/Yapay Zeka/LAB/Ödevler/Python Dosyaları/diyabet.csv")


X = diyabet.iloc[:,:-1]
y = diyabet.clas

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=10)


from sklearn.linear_model import LogisticRegression

mod = LogisticRegression()

mod.fit(X_train,y_train)
y_pred = mod.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

a = accuracy_score(y_test, y_pred)
b = confusion_matrix(y_test, y_pred)
c = f1_score(y_test, y_pred)

print(a)
print(b)
print(c)




