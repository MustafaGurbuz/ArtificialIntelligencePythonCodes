"""
#1.SORU CEVABI
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

veri = pd.read_csv("F:/YAZILIM MÜHENDİSLİĞİ/4.SINIF/Yapay Zeka/LAB/Ödevler/Python Dosyaları/Verisetleri/telefon_fiyat_değişimi.csv")

label_encoder = LabelEncoder().fit(veri.price_range)
labels = label_encoder.transform(veri.price_range)
classes = list(label_encoder.classes_)

X = veri.drop(["price_range"],axis=1)
y=labels

nb_features=20
nb_classes=len(classes)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.5)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

X_train = np.array(X_train).reshape(1000,20,1)
X_test = np.array(X_test).reshape(1000,20,1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Flatten, SimpleRNN, BatchNormalization

model = Sequential()
model.add(Conv1D(1024,1,input_shape=(nb_features,1)))
model.add(Activation("relu"))
model.add(MaxPooling1D(2))
model.add(Conv1D(512,1))
model.add(MaxPooling1D(1))
model.add(Conv1D(128,1))
model.add(MaxPooling1D(2))
model.add(SimpleRNN(1024,input_shape=(nb_features,1)))
model.add(Activation("relu"))
model.add(Dropout(0.30))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(2018,activation="relu"))
model.add(Dense(1024,activation="relu"))
model.add(Dense(512,activation="relu"))
model.add(Dense(256,activation="relu"))
model.add(Dense(nb_classes,activation="sigmoid"))
model.summary()


from tensorflow.keras.optimizers import SGD

opt = SGD(lr=1e-3, decay=1e-3, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer = opt, metrics=["accuracy"])
score = model.fit(X_train,y_train,epochs=100, validation_data=(X_test,y_test))


print("Ortalama Eğitim kaybı: ",np.mean(model.history.history["loss"]))
print("Ortalama Eğitim başarımı: ",np.mean(model.history.history["acc"]))
print("Ortalama Doğrulama kaybı: ",np.mean(model.history.history["val_loss"]))
print("Ortalama Doğrulama başarımı: ",np.mean(model.history.history["val_acc"]))

import matplotlib.pyplot as plt

plt.plot(model.history.history["acc"])
plt.plot(model.history.history["val_acc"])
plt.title("Model Başarımları")
plt.xlabel("Epok Sayısı")
plt.ylabel("Başarım")
plt.legend(["Eğitim","Test"], loc="lower right")
plt.show()

plt.plot(model.history.history["loss"])
plt.plot(model.history.history["val_loss"])
plt.title("Model Kayıpları")
plt.xlabel("Epok Sayısı")
plt.ylabel("Kayıp")
plt.legend(["Eğitim","Test"], loc="upper right")
plt.show()
"""
#--------------------------------- 1.SORU SONU ------------------------------------
"""
#2.SORU CEVABI
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

credit = pd.read_csv("F:/YAZILIM MÜHENDİSLİĞİ/4.SINIF/Yapay Zeka/LAB/Ödevler/Python Dosyaları/Verisetleri/creditcard.csv")

label_encoder = LabelEncoder().fit(credit.Class)
labels = label_encoder.transform(credit.Class)
classes = list(label_encoder.classes_)

X = credit.drop(["Class"],axis=1)
y=labels

nb_features=30
nb_classes=len(classes)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

X_train = np.array(X_train).reshape(199364,30,1)
X_test = np.array(X_test).reshape(85443,30,1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Conv1D,MaxPooling1D
from tensorflow.keras.layers import Flatten, LSTM, BatchNormalization

model = Sequential()
model.add(Conv1D(1024,1,input_shape=(nb_features,1)))
model.add(Activation("relu"))
model.add(MaxPooling1D(2))
model.add(Conv1D(512,1))
model.add(MaxPooling1D(1))
model.add(Conv1D(128,1))
model.add(MaxPooling1D(2))
model.add(LSTM(512,input_shape=(nb_features,1)))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(2018,activation="relu"))
model.add(Dense(1024,activation="relu"))
model.add(Dense(nb_classes,activation="softmax"))
model.summary()

from keras import backend as K

def recall_m(y_true,y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred,0,1)))
  possible_positives = K.sum(K.round(K.clip(y_true,0,1)))
  recall = true_positives / (possible_positives + K.epsilon())
  return recall

def precision_m(y_true,y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred,0,1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred,0,1)))
  precision = true_positives / (predicted_positives + K.epsilon())
  return precision

def f1_m(y_true,y_pred):
  precision= precision_m(y_true,y_pred)
  recall= recall_m(y_true,y_pred)
  return 2*((precision*recall)/(precision+recall+K.epsilon()))

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy",f1_m,precision_m,recall_m])
score=model.fit(X_train,y_train,epochs=50,validation_data=(X_test,y_test))

import matplotlib.pyplot as plt

plt.plot(model.history.history["acc"])
plt.plot(model.history.history["val_acc"])
plt.title("Model Başarımları")
plt.xlabel("Epok Sayısı")
plt.ylabel("Başarım")
plt.legend(["Eğitim","Test"], loc="lower right")
plt.show()

plt.plot(model.history.history["loss"])
plt.plot(model.history.history["val_loss"])
plt.title("Model Kayıpları")
plt.xlabel("Epok Sayısı")
plt.ylabel("Kayıp")
plt.legend(["Eğitim","Test"], loc="upper right")
plt.show()

plt.plot(model.history.history["f1_m"],color="y")
plt.plot(model.history.history["val_f1_m"],color='b')
plt.title("Model F1-Skorları")
plt.xlabel("Epok Sayısı")
plt.ylabel("F1-Skor")
plt.legend(["Eğitim","Test"], loc="upper right")
plt.show()

plt.plot(model.history.history["precision_m"],color="y")
plt.plot(model.history.history["val_precision_m"],color='b')
plt.title("Model Kesinlik")
plt.xlabel("Epok Sayısı")
plt.ylabel("Kesinlik")
plt.legend(["Eğitim","Test"], loc="upper right")
plt.show()

plt.plot(model.history.history["recall_m"],color="y")
plt.plot(model.history.history["val_recall_m"],color='b')
plt.title("Model Duyarlılık")
plt.xlabel("Epok Sayısı")
plt.ylabel("Duyarlılık")
plt.legend(["Eğitim","Test"], loc="upper right")
plt.show()

#------------------------- AUC EĞRİSİ -------------------------------- 
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize

X_train = np.array(X_train).reshape(199364,30)
X_test = np.array(X_test).reshape(85443,30)

# classifier
clf = OneVsRestClassifier(LinearSVC(random_state=0))
y_score = clf.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for a specific class
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i],color='m')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC example')
    plt.legend(loc="lower right")
    plt.show()
"""
#--------------------------- 2.SORU SONU ----------------------------------------
"""
#3.SORU CEVABI
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

heart = pd.read_csv("F:/YAZILIM MÜHENDİSLİĞİ/4.SINIF/Yapay Zeka/LAB/Ödevler/Python Dosyaları/Verisetleri/heart.csv")

label_encoder = LabelEncoder().fit(heart.target)
labels = label_encoder.transform(heart.target)
classes = list(label_encoder.classes_)

X = heart.drop(["target"],axis=1)
y=labels

nb_features=13
nb_classes=len(classes)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

X_train = np.array(X_train).reshape(181,13,1)
X_test = np.array(X_test).reshape(122,13,1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Conv1D,MaxPooling1D
from tensorflow.keras.layers import Flatten, LSTM, BatchNormalization

model = Sequential()
model.add(Conv1D(2048,1,input_shape=(nb_features,1)))
model.add(Activation("relu"))
model.add(MaxPooling1D(2))
model.add(Conv1D(1024,1))
model.add(MaxPooling1D(1))
model.add(Conv1D(256,1))
model.add(MaxPooling1D(1))
model.add(Conv1D(128,1))
model.add(MaxPooling1D(2))
model.add(LSTM(512,input_shape=(nb_features,1)))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(2018,activation="relu"))
model.add(Dense(1024,activation="relu"))
model.add(Dense(1024,activation="relu"))
model.add(Dense(nb_classes,activation="softmax"))
model.summary()

from keras import backend as K

def recall_m(y_true,y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred,0,1)))
  possible_positives = K.sum(K.round(K.clip(y_true,0,1)))
  recall = true_positives / (possible_positives + K.epsilon())
  return recall

def precision_m(y_true,y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred,0,1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred,0,1)))
  precision = true_positives / (predicted_positives + K.epsilon())
  return precision

def f1_m(y_true,y_pred):
  precision= precision_m(y_true,y_pred)
  recall= recall_m(y_true,y_pred)
  return 2*((precision*recall)/(precision+recall+K.epsilon()))

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy",f1_m,precision_m,recall_m])
score=model.fit(X_train,y_train,epochs=200,validation_data=(X_test,y_test))

import matplotlib.pyplot as plt

plt.plot(model.history.history["acc"])
plt.plot(model.history.history["val_acc"])
plt.title("Model Başarımları")
plt.xlabel("Epok Sayısı")
plt.ylabel("Başarım")
plt.legend(["Eğitim","Test"], loc="lower right")
plt.show()

plt.plot(model.history.history["loss"])
plt.plot(model.history.history["val_loss"])
plt.title("Model Kayıpları")
plt.xlabel("Epok Sayısı")
plt.ylabel("Kayıp")
plt.legend(["Eğitim","Test"], loc="upper right")
plt.show()

plt.plot(model.history.history["f1_m"],color="y")
plt.plot(model.history.history["val_f1_m"],color='b')
plt.title("Model F1-Skorları")
plt.xlabel("Epok Sayısı")
plt.ylabel("F1-Skor")
plt.legend(["Eğitim","Test"], loc="upper right")
plt.show()

plt.plot(model.history.history["precision_m"],color="g")
plt.plot(model.history.history["val_precision_m"],color='r')
plt.title("Model Kesinlik")
plt.xlabel("Epok Sayısı")
plt.ylabel("Kesinlik")
plt.legend(["Eğitim","Test"], loc="upper right")
plt.show()

plt.plot(model.history.history["recall_m"],color="k")
plt.plot(model.history.history["val_recall_m"],color='m')
plt.title("Model Duyarlılık")
plt.xlabel("Epok Sayısı")
plt.ylabel("Duyarlılık")
plt.legend(["Eğitim","Test"], loc="upper right")
plt.show()

#------------------------- AUC EĞRİSİ -------------------------------- 
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize

X_train = np.array(X_train).reshape(181,13)
X_test = np.array(X_test).reshape(122,13)

# classifier
clf = OneVsRestClassifier(LinearSVC(random_state=0))
y_score = clf.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for a specific class
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i],color='r')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC example')
    plt.legend(loc="lower right")
    plt.show()
"""
#----------------------------- SON SORU CEVABI -------------------------------
