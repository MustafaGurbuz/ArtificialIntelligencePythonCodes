"""
#1.SORU CEVABI
carpimTablosu = int(input("Bir değer giriniz: "))
for i in range(1,11):
    c=i*carpimTablosu
    print(c)
"""

"""
#2.SORU CEVABI
sayi=int(input("Bir sayı giriniz: "))
count=0
sart=True
while sart:
   c=sayi%10
   sayi=(sayi-c)/10
   count+=1
   if sayi==0:
       sart=False
print("Basamak Sayısı: ",count)
"""

"""
#3.SORU CEVABI
sayısalDegerler = [12,15,32,42,55,75,122,132,150,180,200]
for i in range(11):
    if sayısalDegerler[i]<=150:
        if sayısalDegerler[i]%5==0:
            print(sayısalDegerler[i])

count=0
while count<11:
     if sayısalDegerler[count]<=150:
        if sayısalDegerler[count]%5==0:
            print(sayısalDegerler[count])
        count+=1
"""

"""
#4.SORU CEVABI
a=int(input("Birinci degeri giriniz: "))
b=int(input("İkinci degeri giriniz: "))
c=int(input("Üçüncü degeri giriniz: "))

for i in range(a,b+1):
    if i%c==0:
        print(i)
"""

"""
#5.SORU CEVABI
i=0
j=100
while (i<99 or j>1):
    i+=1
    j-=1
    print(i,j)
"""  

"""
#6.SORU CEVABI
ip=[int(input("ip1: ")),int(input("ip2: ")),int(input("ip3: ")),int(input("ip4: "))]
for i in range(5):
    if ip[0] <= 255 and ip[1] <= 255 and ip[2] <= 255 and ip[3] <= 255:
        ip[3]+=1
        if ip[3]>255: 
            ip[3]=0
            ip[2]=0
            if ip[1]<255:
                ip[1]+=1
            else:
                ip[1]=0
                if ip[0]<255:
                    ip[0]+=1
                else:
                    ip[0]=0
        print(ip)
"""