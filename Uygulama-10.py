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
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

X_train = np.array(X_train).reshape(1400,20,1)
X_test = np.array(X_test).reshape(600,20,1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation
from tensorflow.keras.layers import Flatten, LSTM, BatchNormalization

model = Sequential()
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
score=model.fit(X_train,y_train,epochs=100,validation_data=(X_test,y_test))

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
plt.title("Model Kayıpları")
plt.xlabel("Epok Sayısı")
plt.ylabel("Kayıp")
plt.legend(["Eğitim","Test"], loc="upper right")
plt.show()
"""
#-------------------------------------------------------------------------------------------------------
"""
#2.SORU CEVABI
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split
import os

print(os.listdir("F:/YAZILIM MÜHENDİSLİĞİ/4.SINIF/Yapay Zeka/LAB/Ödevler/Python Dosyaları/Verisetleri/Kaktüs"))

ImageWidth = 75
ImageHeight = 75
ImageSize = (ImageWidth, ImageHeight)
ImageChannels = 3

fileNames = os.listdir("F:/YAZILIM MÜHENDİSLİĞİ/4.SINIF/Yapay Zeka/LAB/Ödevler/Python Dosyaları/Verisetleri/Kaktüs/train")
categories = []
for filename in fileNames:
    category = filename.split('1')[0]
    if (category == 'a' or category == 'b' or category == 'c' or category == 'd' or 
        category == 'e' or category == 'f' or category == 'g' or category == 'h' or 
        category == 'j' or category == 'k' or category == 'l' or category == 'm' or
        category == 'n' or category == 'o' or category == 'p' or category == 'r' or
        category == 's' or category == 't' or category == 'q' or category == 'u' or
        category == 'v' or category == 'w' or category == 'y' or category == 'z' or 
        category=='0' or category=='1' or category == '2' or category =='3'):
            categories.append(1)
    else:
            categories.append(0)
df = pd.DataFrame({
        'filename' : fileNames,
        'category' : categories
    })

df["category"].value_counts().plot.bar()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation
from tensorflow.keras.layers import Flatten, LSTM, BatchNormalization

model = Sequential()
model.add(LSTM(512,batch_input_shape=(ImageWidth,ImageHeight, ImageChannels)))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(2018,activation="relu"))
model.add(Dense(1024,activation="relu"))
model.add(Dense(2,activation="softmax"))
model.summary()

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

df["category"] = df["category"].replace({0: "no", 1: "yes"})
train_df, validate_df = train_test_split(df,test_size=0.2)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

train_df["category"].value_counts().plot.bar()

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size = 15


train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
   "F:/YAZILIM MÜHENDİSLİĞİ/4.SINIF/Yapay Zeka/LAB/Ödevler/Python Dosyaları/Verisetleri/Kaktüs/train/",
    x_col='filename',
    y_col='category',
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
   "F:/YAZILIM MÜHENDİSLİĞİ/4.SINIF/Yapay Zeka/LAB/Ödevler/Python Dosyaları/Verisetleri/Kaktüs/train/",
    x_col='filename',
    y_col='category',
   )


epochs=75
history=model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate,
    steps_per_epoch=total_train
    )


test_filenames=os.listdir("F:/YAZILIM MÜHENDİSLİĞİ/4.SINIF/Yapay Zeka/LAB/Ödevler/Python Dosyaları/Verisetleri/Kaktüs/test/")
test_df=pd.DataFrame({
    'filename': test_filenames
    })
nb_samples=test_df.shape[0]

test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df,
    "F:/YAZILIM MÜHENDİSLİĞİ/4.SINIF/Yapay Zeka/LAB/Ödevler/Python Dosyaları/Verisetleri/Kaktüs/test",
    x_col='filename',
    y_col=None,
    class_mode=None,
    batch_size=batch_size,
    shuffle=False
)

predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))

test_df['category']=np.argmax(predict, axis=-1)
label_map= dict((v, k) for k,v in train_generator.class_indices.items())
test_df['category'] = test_df['category'].replace(label_map)

import matplotlib.pyplot as plt

sample_test = test_df.head(30)
sample_test.head()
plt.figure(figsize=(6,12))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img("F:/YAZILIM MÜHENDİSLİĞİ/4.SINIF/Yapay Zeka/LAB/Ödevler/Python Dosyaları/Verisetleri/Kaktüs/test/"+
                       filename)
    plt.subplot(6, 5, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')')
plt.tight_layout()
plt.show()

print("Ortalama Eğitim kaybı: ",np.mean(model.history.history["loss"]))
print("Ortalama Eğitim başarımı: ",np.mean(model.history.history["acc"]))
"""
#--------------------------------------------------------------------------------------------
"""
#3.SORU CEVABI
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

veri = pd.read_csv("F:/YAZILIM MÜHENDİSLİĞİ/4.SINIF/Yapay Zeka/LAB/Ödevler/Python Dosyaları/Verisetleri/digit-recognizer/train.csv")
test = pd.read_csv("F:/YAZILIM MÜHENDİSLİĞİ/4.SINIF/Yapay Zeka/LAB/Ödevler/Python Dosyaları/Verisetleri/digit-recognizer/test.csv")

label_encoder = LabelEncoder().fit(veri.label)
labels = label_encoder.transform(veri.label)
classes = list(label_encoder.classes_)

X = veri.drop(["label"],axis=1)
y=labels

nb_features=784
nb_classes=len(classes)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

X_train = np.array(X_train).reshape(29400, 784,1)
X_test = np.array(X_test).reshape(12600, 784,1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation
from tensorflow.keras.layers import Flatten, LSTM, BatchNormalization

model = Sequential()
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
score=model.fit(X_train,y_train,epochs=100,validation_data=(X_test,y_test))

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
"""
