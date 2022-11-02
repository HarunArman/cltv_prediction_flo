import pandas as pd
import datetime as dt


df = pd.read_csv("flo_data_20k.csv")


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


df.head(10)
df.columns
df.shape
df.describe().T
df.describe(include="O").T
df.isnull().sum()
df.info()


df["omnichannel_purchases"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["omnichannel_total_price"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]


df.info()

convert_date = [i for i in df.columns if "date" in i]

for i in convert_date:
    df[i] = pd.to_datetime(df[i])

df.info()


df.groupby("order_channel").agg({"omnichannel_total_price": "sum",
                                 "omnichannel_purchases": "sum",
                                 "order_channel": lambda x: x.value_counts()})


df.sort_values(by="omnichannel_total_price", ascending=False).head(10)
df.sort_values(by="omnichannel_purchases", ascending=False).head(10)


def data_prep(dataframe):
    dataframe["omnichannel_purchases"] = dataframe["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    dataframe["omnichannel_total_price"] = dataframe["customer_value_total_ever_online"] + df[
        "customer_value_total_ever_offline"]

    convert_date = [i for i in dataframe.columns if "date" in i]
    for i in convert_date:
        dataframe[i] = pd.to_datetime(dataframe[i])

    return dataframe


df = data_prep(df)

df.head()


df["last_order_date"].max()

today_date = dt.datetime(2021, 6, 2)

rfm = df.groupby("master_id").agg({"last_order_date": lambda x: (today_date - x.max()).days,
                                   "omnichannel_purchases": lambda x: x,
                                   "omnichannel_total_price": lambda x: x})

rfm.reset_index(inplace=True)


rfm.columns = ["master_id", "recency", "frequency", "monetary"]

rfm.head()


rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm["rf_score"] = (rfm['recency_score'].astype(str) +
                   rfm['frequency_score'].astype(str))


seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['rf_score'].replace(seg_map, regex=True)


rfm.groupby("segment").agg({"recency": "mean",
                            "frequency": "mean",
                            "monetary": "mean"})


rfm["categories"] = df["interested_in_categories_12"]

target_df = rfm[(rfm["segment"] == "champions") | (rfm["segment"] == "loyal_customers")
                & (rfm["categories"].str.contains("KADIN"))]["master_id"]

target_df.to_csv("targets.csv", index=False)


target_b_df = rfm[((rfm["segment"] == "about_to_sleep")
                   | (rfm["segment"] == "cant_loose")
                   | (rfm["segment"] == "new_customers"))
                  & ((rfm["categories"].str.contains("ERKEK"))
                     | (rfm["categories"].str.contains("COCUK")))]

target_b_df.to_csv("targets_b.csv", index=False)
