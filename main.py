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

df_ = pd.read_csv("datasets/data.csv",encoding="ISO-8859-1")
df = df_.copy()
df.head()
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


df.head()
df.describe().T
df.isnull().sum()
df.info()
df["InvoiceNo"].unique()
# customer id ve descriptionların boş olmasıverimizin sonuçlarını saptırabilir o yğzden boş değerleri siliyorum

df.dropna(inplace=True)

# başında C olan invoice no' lar genelde iadeler. iadeleri istemiyoruz
df = df[~df["InvoiceNo"].str.contains("C", na=False)]

# invoice date' i tarih yapmamız gerek
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# quantity ve unit price'ların 0'dan büyük olmasını istiyoruz
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]




#aykırı değerleri baskılarız

replace_with_thresholds(df,"UnitPrice")
replace_with_thresholds(df,"Quantity")

# aykırı değerleri baskıladıktan sonra standart sapmanın ve max değerin düştüğünü gözlemlemekteyiz


# alınan ürünün fiyatını hespalarız
df["TotalPrice"] = df["UnitPrice"] * df["Quantity"]



# CLTV YAPISI

# analiz gününü hesaplayalım bu bize recency ve T değerleri için gerekecek:
df["InvoiceDate"].max()
# bu tarihe iki gün ekleyeceğiz neden :
analysis_date = dt.datetime(2011,12,11)



# recency , T , frequency ve monetary değerleri lazım
# recency değeri , rfm'deki recency değerinden farklı olarak, müşterinin
#Son satın alma ile ilk satın alma arasındaki süre
# T : müşterinin yaşı ,haftalık bazda,
#fequency müşterinin ne sıklıkla alışveriş yaptığı, fatura sayısı
#monetary : rfm den farklı olarak satın alma başına ortalama kazançtır
cltv_df = df.groupby("CustomerID").agg({"InvoiceDate" :[ lambda InvoiceDate : (InvoiceDate.max() - InvoiceDate.min()).days,
                                                 lambda InvoiceDate : (analysis_date - InvoiceDate.min()).days],
                               "InvoiceNo": lambda InvoiceNo : InvoiceNo.nunique(),
                               "TotalPrice" : lambda TotalPrice : TotalPrice.sum()})



cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ["recency_cltv_weekly","T_weekly","frequency","monetary_cltv_avg"]
cltv_df = cltv_df[cltv_df["frequency"] > 1]

# haftalık cinsten belirtelim  :
cltv_df["recency_cltv_weekly"] = cltv_df["recency_cltv_weekly"] / 7
cltv_df["T_weekly"] = cltv_df["T_weekly"] /7

# monetary ise satın alma başına ortalama değer
cltv_df["monetary_cltv_avg"] = cltv_df["monetary_cltv_avg"]  / cltv_df["frequency"]