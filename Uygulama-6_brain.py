"""
#SORU 4
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split
import os
import string

print(os.listdir("F:/YAZILIM MÜHENDİSLİĞİ/4.SINIF/Yapay Zeka/LAB/Ödevler/Python Dosyaları/beyin_tümörü"))

ImageWidth = 75
ImageHeight = 75
ImageSize = (ImageWidth, ImageHeight)
ImageChannels = 3


fileNames = os.listdir("F:/YAZILIM MÜHENDİSLİĞİ/4.SINIF/Yapay Zeka/LAB/Ödevler/Python Dosyaları/beyin_tümörü/evet")
categories = []
for filename in fileNames:
    category = filename.split(' ')[0]
    if category == 'yes':
        categories.append(1)
    else:
        categories.append(0)
df = pd.DataFrame({
        'filename' : fileNames,
        'category' : categories
    })

df["category"].value_counts().plot.bar()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization

model = Sequential()
model.add(Conv2D(32, (3, 3), activation ='relu',input_shape=(ImageWidth,ImageHeight, ImageChannels)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

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
    "F:/YAZILIM MÜHENDİSLİĞİ/4.SINIF/Yapay Zeka/LAB/Ödevler/Python Dosyaları/beyin_tümörü/evet/",
    x_col='filename',
    y_col='category',
    target_size=ImageSize,
    class_mode='categorical',
    batch_size=batch_size
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    "F:/YAZILIM MÜHENDİSLİĞİ/4.SINIF/Yapay Zeka/LAB/Ödevler/Python Dosyaları/beyin_tümörü/evet/",
    x_col='filename',
    y_col='category',
    target_size=ImageSize,
    class_mode='categorical',
    batch_size=batch_size
)


epochs=75
history=model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate,
    steps_per_epoch=total_train
    )


test_filenames=os.listdir("F:/YAZILIM MÜHENDİSLİĞİ/4.SINIF/Yapay Zeka/LAB/Ödevler/Python Dosyaları/beyin_tümörü/hayır/")
test_df=pd.DataFrame({
    'filename': test_filenames
    })
nb_samples=test_df.shape[0]

test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df,
    "F:/YAZILIM MÜHENDİSLİĞİ/4.SINIF/Yapay Zeka/LAB/Ödevler/Python Dosyaları/beyin_tümörü/hayır/",
    x_col='filename',
    y_col=None,
    target_size=ImageSize,
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
    img = load_img("F:/YAZILIM MÜHENDİSLİĞİ/4.SINIF/Yapay Zeka/LAB/Ödevler/Python Dosyaları/beyin_tümörü/hayır/"+
                       filename, target_size=ImageSize)
    plt.subplot(6, 5, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')')
plt.tight_layout()
plt.show()

print("Ortalama Eğitim kaybı: ",np.mean(model.history.history["loss"]))
print("Ortalama Eğitim başarımı: ",np.mean(model.history.history["acc"]))
"""