"""
//1.SORU CEVABI
birinciAci = int(input("1.açıyı giriniz: "))
ikinciAci = int(input("2.açıyı giriniz: "))
ucuncuAci = int(input("3.açıyı giriniz: "))

toplam = birinciAci + ikinciAci + ucuncuAci
if (birinciAci > 90 or ikinciAci > 90 or ucuncuAci > 90):
    if toplam == 180:
        print("Geniş Açılı Üçgen")
    else:
        print("Girdiğiniz değerler olması gerekenden fazla")
elif (birinciAci == 90 or ikinciAci == 90 or ucuncuAci == 90):
    if toplam == 180:
        print("Dik Açılı Üçgen")
    else:
        print("Girdiğiniz değerler olması gerekenden fazla")
elif (birinciAci < 90 or ikinciAci < 90 or ucuncuAci < 90):
    if toplam == 180:
        print("Dar Açılı Üçgen")
    else:
        print("Girdiğiniz değerler olması gerekenden fazla")
else:
    print("Girdiğiniz değerler yanlış")
"""

"""
// 2.SORU CEVABI
print("Renkler : Yeşil(yeşil) | Sarı(sarı) | Kırmızı(kırmızı)")
uzayli_rengi = str(input("Bir Renk Giriniz : "))
if (uzayli_rengi == "Yeşil" or uzayli_rengi == "yeşil" or 
    uzayli_rengi == "Sarı" or  uzayli_rengi == "sarı" or
    uzayli_rengi == "Kırmızı" or uzayli_rengi == "kırmızı"):
    if uzayli_rengi == "Yeşil" or uzayli_rengi == "yeşil":
        print("Tebrikler,yeşil uzaylıya ateş ettiğiniz için 5 puan kazandınız.")
    else:
        print("Tebrikler,yeşil olmayan uzaylıya ateş ettiğiniz için 10 puan kazandınız.")
else:
    print("Yanlış renk girdiniz")
"""

"""
//3.SORU CEVABI
print("Renkler : Yeşil(yeşil) | Sarı(sarı) | Kırmızı(kırmızı)")
uzayli_rengi = str(input("Bir Renk Giriniz : "))
if (uzayli_rengi == "Yeşil" or uzayli_rengi == "yeşil" or 
    uzayli_rengi == "Sarı" or  uzayli_rengi == "sarı" or
    uzayli_rengi == "Kırmızı" or uzayli_rengi == "kırmızı"):
    if uzayli_rengi == "Yeşil" or uzayli_rengi == "yeşil":
        print("Tebrikler,yeşil uzaylıya ateş ettiğiniz için 5 puan kazandınız.")
    elif uzayli_rengi == "Sarı" or uzayli_rengi == "sarı":
        print("Tebrikler,sarı uzaylıya ateş ettiğiniz için 10 puan kazandınız.")
    else:
        print("Tebrikler,kırmızı uzaylıya ateş ettiğiniz için 15 puan kazandınız.")
else:
    print("Yanlış renk girdiniz")
"""

"""
//4.SORU CEVABI
yas = int(input("Bir Yaş Değeri Giriniz: "))
if yas >= 0:
    if yas < 2:
        print("Bu kişi bebektir")
    elif yas >= 2 and yas < 4:
        print("Bu kişi yeni yürümeye başlayan çocuktur")
    elif yas >= 4 and yas < 13:
        print("Bu kişi çocuktur")
    elif yas >= 13 and yas < 20:
        print("Bu kişi ergendir")
    elif yas >=20 and yas < 65:
        print("Bu kişi yetişkindir")
    else:
        print("Bu kişi yaşlıdır")
else:
    print("Girdiğiniz değer yanlıştır.")
"""


"""
//5.SORU CEVABI
favori_meyveler = ['Elma','Armut','Kiraz','Muz','Çilek']
örnek_meyveler = ['Elma','Armut','Karpuz','Kavun','Muz',
                  'Portakal','Çilek','Vişne','Kiraz','Mandalina']

print(favori_meyveler)
print(örnek_meyveler)

if (favori_meyveler[0] == örnek_meyveler[0] and favori_meyveler[1] == örnek_meyveler[1] 
    and favori_meyveler[3] == örnek_meyveler[4] and favori_meyveler[4] == örnek_meyveler[6] and
    favori_meyveler[2] == örnek_meyveler[8]):
    print("Favori meyveler, örnek meyveler içerisinde yer alıyor")
else:
    print("Favori meyveler, örnek meyveler içerisinde yer almıyor")
"""