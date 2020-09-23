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


