import pandas as pd
import numpy as np

#Veri Okuma(Soru 1)
veri = pd.read_csv("C:/Users/MUSTAFA/Desktop/oyun.csv")
#print(veri)

#Veri Silme(Soru 2)
sütunSil = veri.drop(['Sıcaklık','Nem'],axis=1)
#print(sütunSil)

#DataFrame(Soru 3)
oyun_data = {
'Gün' : ['G1','G2','G3','G4','G5','G6','G7','G8','G9',
'G10','G11','G12','G13','G14'],
'Hava Durumu' : ['Güneşli','Güneşli','Kapalı','Yağmurlu','Yağmurlu','Yağmurlu',
'Kapalı','Güneşli','Güneşli','Yağmurlu','Güneşli','Kapalı','Kapalı','Yağmurlu'
],
'Sıcaklık' : ['Sıcak','Sıcak','Sıcak','Ilıman','Soğuk','Soğuk','Soğuk',
'Ilıman','Soğuk','Ilıman','Ilıman','Ilıman','Sıcak','Ilıman'
],
'Nem' : ['Yüksek','Yüksek','Yüksek','Yüksek','Normal','Normal','Normal','Yüksek',
         'Normal','Normal','Normal','Yüksek','Normal','Yüksek'],
'Yağış' : ['Seyrek','Aşırı','Seyrek','Seyrek','Seyrek','Aşırı','Aşırı',
           'Seyrek','Seyrek','Seyrek','Aşırı','Aşırı','Seyrek','Aşırı'],
'Oyun' : ['Yok','Yok','Var','Var','Var','Yok','Var','Var','Yok','Var','Var',
          'Yok','Var','Yok']
}
df = pd.DataFrame(oyun_data)
#print(df)

#Betimleyici Bilgiler(Soru 3)
"""
df.describe()
Out[38]: 
        Gün Hava Durumu Sıcaklık     Nem   Yağış Oyun
count    14          14       14      14      14   14
unique   14           3        3       2       2    2
top     G13    Yağmurlu   Ilıman  Normal  Seyrek  Var
freq      1           5        6       7       8    8
"""

#Dizi Oluşturma(Soru 4)(3,4)
a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
#print(a)

#(Soru 4)(6,2)
at = a.reshape(6,2)
#print(at)

#Rastgele Dizi oluşturma ve Stack(Soru 5)
da = np.random.randint(1,10,size=(3,3))
db = np.random.randint(10,100,size=(3,3))
#print(da)
print('\n')
#print(db)
print('\n')
sta = np.stack((da,db),axis = 1)
#print(sta)