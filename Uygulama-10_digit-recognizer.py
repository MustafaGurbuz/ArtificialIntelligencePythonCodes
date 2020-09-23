import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


veri = pd.read_csv("F:/YAZILIM MÜHENDİSLİĞİ/4.SINIF/Yapay Zeka/LAB/Ödevler/Python Dosyaları/Verisetleri/digit-recognizer/train.csv")
test = pd.read_csv("F:/YAZILIM MÜHENDİSLİĞİ/4.SINIF/Yapay Zeka/LAB/Ödevler/Python Dosyaları/Verisetleri/digit-recognizer/test.csv")

model = TSNE(n_components=2, random_state=0)

tsne_data = model.fit_transform(veri)
