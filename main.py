import pandas as pd
import datetime as dt

from lifetimes import BetaGeoFitter, GammaGammaFitter

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


df = pd.read_csv("flo_data_20k.csv")


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return round(low_limit), round(up_limit)


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


col_list = ["order_num_total_ever_online", "order_num_total_ever_offline",
            "customer_value_total_ever_offline", "customer_value_total_ever_online"]

for i in col_list:
    replace_with_thresholds(df,i)


df["omnichannel_purchases"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["omnichannel_total_price"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]


convert_date = [i for i in df.columns if "date" in i]

for i in convert_date:
    df[i] = pd.to_datetime(df[i])

df.info()


df["last_order_date"].max()

today_date = dt.datetime(2021, 6, 2)


cltv_df = pd.DataFrame({"customer_id": df["master_id"],
             "recency_cltv_weekly": (df["last_order_date"] - df["first_order_date"]).dt.days / 7,
             "T_weekly": (today_date - df["first_order_date"]).dt.days / 7,
             "frequency": df["omnichannel_purchases"],
             "monetary_cltv_avg": df["omnichannel_total_price"] / df["omnichannel_purchases"]})

cltv_df.head()


bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])


cltv_df["exp_sales_3_month"] = bgf.predict(4*3, cltv_df['frequency'],
                                           cltv_df['recency_cltv_weekly'],
                                           cltv_df['T_weekly'])


cltv_df["exp_sales_6_month"] = bgf.predict(4*6, cltv_df['frequency'],
                                           cltv_df['recency_cltv_weekly'],
                                           cltv_df['T_weekly'])


ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                       cltv_df['monetary_cltv_avg'])


cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv = cltv.reset_index(drop=True)

cltv_df["cltv_6_months"] = cltv

cltv_df.sort_values("cltv_6_months", ascending=False)[0:20]


cltv_df["segment"] = pd.qcut(cltv_df["cltv_6_months"], 4, labels=["D", "C", "B", "A"])
cltv_df.groupby("segment").agg(
    {"count", "mean", "sum"})


