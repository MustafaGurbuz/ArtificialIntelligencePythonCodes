"""
#1,2 ve 3. SORU CEVAPLARI
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

veri = pd.read_csv("F:/YAZILIM MÜHENDİSLİĞİ/4.SINIF/Yapay Zeka/LAB/Ödevler/Python Dosyaları/telefon_fiyat_değişimi.csv")

label_encoder = LabelEncoder().fit(veri.price_range)
labels = label_encoder.transform(veri.price_range)
classes = list(label_encoder.classes_)

#3.SORU CEVABI
X = veri.drop(["price_range","blue","fc","int_memory","ram","wifi"],axis=1)
y=labels

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.SORU CEVABI
model = Sequential()
model.add(Dense(12,input_dim=15,activation="relu"))
model.add(Dense(7,activation="relu"))
model.add(Dense(5,activation="relu"))
model.add(Dense(4,activation="softmax"))
model.summary()

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=150)


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

#2.SORU CEVABI
from sklearn.model_selection import cross_val_score
from sklearn import svm
clf = svm.SVC(kernel="linear",C=1)
scores=cross_val_score(clf,X,y,cv=5,scoring="accuracy")
plt.plot(model.history.history["acc"])
plt.title("Model Başarımları")
plt.xlabel("Epok Sayısı")
plt.ylabel("Başarım")
"""

"""
#4.SORU CEVABI
import pandas as pd
from sklearn.preprocessing import LabelEncoder

diyabet = pd.read_csv("F:/YAZILIM MÜHENDİSLİĞİ/4.SINIF/Yapay Zeka/LAB/Ödevler/Python Dosyaları/diyabet.csv")

le = LabelEncoder().fit(diyabet.clas)
labs = le.transform(diyabet.clas)
classes = list(le.classes_)

A=diyabet.drop(["clas"],axis=1)
b=labs

from sklearn.preprocessing import StandardScaler

sa = StandardScaler()
A = sa.fit_transform(A)

from sklearn.model_selection import train_test_split
A_train, A_test, b_train, b_test = train_test_split(A,b, test_size=0.1,random_state=2)


from tensorflow.keras.utils import to_categorical
b_train = to_categorical(b_train)
b_test = to_categorical(b_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

mod = Sequential()
mod.add(Dense(7.2,input_dim=8,activation="relu"))
mod.add(Dense(12,activation="relu"))
mod.add(Dense(2,activation="softmax"))
mod.summary()

mod.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
mod.fit(A_train,b_train,validation_data=(A_test,b_test),epochs=150)


import matplotlib.pyplot as plt

plt.plot(mod.history.history["acc"])
plt.plot(mod.history.history["val_acc"])
plt.title("Model Başarımları")
plt.xlabel("Epok Sayısı")
plt.ylabel("Başarım")
plt.legend(["Eğitim","Test"], loc="lower right")
plt.show()

plt.plot(mod.history.history["loss"])
plt.plot(mod.history.history["val_loss"])
plt.title("Model Kayıpları")
plt.xlabel("Epok Sayısı")
plt.ylabel("Kayıp")
plt.legend(["Eğitim","Test"], loc="upper right")
plt.show()
"""

"""
#5.SORU CEVABI
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.preprocessing import LabelEncoder

diyabet = pd.read_csv("F:/YAZILIM MÜHENDİSLİĞİ/4.SINIF/Yapay Zeka/LAB/Ödevler/Python Dosyaları/diyabet.csv")

le = LabelEncoder().fit(diyabet.clas)
labs = le.transform(diyabet.clas)
classes = list(le.classes_)

X=diyabet.drop(["clas"],axis=1)
y=labs

y = label_binarize(y, classes=[0,1])
n_classes = 2

from sklearn.preprocessing import StandardScaler

sa = StandardScaler()
X = sa.fit_transform(X)

# shuffle and split training and test sets
X_train, X_test, y_train, y_test =\
    train_test_split(X, y, test_size=0.33, random_state=0)
    
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
    
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

mod = Sequential()
mod.add(Dense(7.2,input_dim=8,activation="relu"))
mod.add(Dense(12,activation="relu"))
mod.add(Dense(2,activation="softmax"))
mod.summary()

mod.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
mod.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=150)

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
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC example')
    plt.legend(loc="lower right")
    plt.show()
"""