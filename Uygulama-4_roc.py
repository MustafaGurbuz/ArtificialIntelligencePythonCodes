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