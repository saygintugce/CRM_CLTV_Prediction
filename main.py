import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

from sklearn.preprocessing import MinMaxScaler

# df_'yi orijinal veri olarak okuyup, df = df_.copy() ile bir kopyasını almak,
# özellikle veriyle analiz veya dönüşüm yaparken orijinal veriyi korumak için kullanılan iyi bir pratiktir.

df_ = pd.read_csv("datasets/customer_shopping_data.csv")
df = df_.copy()

# aykırı değerlerimiz var mı diye bakmamız gerekir. bunu yapmamızın sebebi veride normal dağılımdan
# sapma gösteren uç noktaları tespit edip, bunların analiz sonuçlarını yanıltmasını önlemektir.

#burada üst ve alt sınırlarımızı belirledik
#Bu fonksiyon, ilgili değişken için %1 ve %99'luk çeyrek değerleri alarak aykırı değer sınırlarını hesaplar.
#Genellikle %25-%75 (Q1–Q3) aralığı kullanılır, ancak bu örnekte uç değerleri daha geniş yakalayabilmek adına %1–%99 aralığı tercih edilmiştir.

def outlier_thresholds(dataframe,variable):
    quantile1 = dataframe[variable].quantile(0.01)
    quantile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quantile3 - quantile1
    up_limit = quantile3 + 1.5 * interquantile_range
    low_limit = quantile1 - 1.5 * interquantile_range
    return up_limit,low_limit

# burada eğer alt limit veya üst limitten farklı aykırı değer var ise,
# düşükse alt limite, yüksekse üst limite eşitleyecektir.
def replace_with_thresholds(dataframe, variable):
    up_limit, low_limit = outlier_thresholds(dataframe,variable)
    dataframe.loc[(dataframe[variable] < low_limit),variable] = round(low_limit,2)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit,2)
    dataframe[variable] = dataframe[variable].round(2)
# # Float olarak bırakır, sadece virgülden sonra 2 basamak yuvarlar
#round(2) demek, örneğin 1500.45678 sayısını 1500.46 yapar ama veri tipi float olarak kalır.


