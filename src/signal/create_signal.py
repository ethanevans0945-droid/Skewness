import polars as pl
import datetime as dt
import sf_quant.data as sfd
import polars_ols
import statsmodels.api as sm
import pandas as pd



start = dt.date(1987, 1, 1)
end = dt.date(2023, 12, 31)
signal_name = "IS"
price_filter = 5
IC = 0.05
# print(sfd.get_assets_columns())

data = sfd.load_assets(
    start=start,
    end=end,
    columns=[
        "date",
        "barrid",
        "price",
        "return",
        "specific_risk",
        "predicted_beta",
        "market_cap",
        "daily_volume", 
        "specific_return"
    ],
    in_universe=True,
).with_columns(pl.col("return",  "specific_return", "specific_risk").truediv(100))

# T=5
# days=T*21

df = data.sort(["barrid", "date"])


# calculates turnover 
df = df.with_columns(
    (pl.col("daily_volume") / pl.col("market_cap")).alias("turnover")
)

# Get last trading day of each month per stock
df = df.with_columns([
    pl.col("date").cast(pl.Datetime).alias("date_dt"),])

# Create month indicator
df = df.with_columns(
    (pl.col("date_dt").dt.year() * 100 + pl.col("date_dt").dt.month()).alias("yyyymm")
)

# does dummy variable for marketcap
df = df.with_columns(
    pl.col("market_cap")
      .rank(method="average")
      .over("date")
      .alias("rank")
).with_columns(
    (pl.col("rank") /
     pl.col("rank").max().over("date") * 3)
    .ceil()
    .alias("mktcap_bin"))

df = df.with_columns(
    pl.col("mktcap_bin").cast(pl.Utf8)
).to_dummies("mktcap_bin")


df = df.with_columns(pl.col('mktcap_bin_3.0').alias('curmktb3'))
df = df.with_columns(pl.col('mktcap_bin_2.0').alias('curmktb2'))
df = df.with_columns(pl.col('mktcap_bin_1.0').alias('curmktb1'))
df = df.drop(['rank', 'mktcap_bin_3.0', 'mktcap_bin_2.0', 'mktcap_bin_1.0'])
# df = df.with_columns(
#     pl.col("market_cap")
#       .qcut(3)
#       .over("date")
#       .cast(pl.Utf8)
#       .alias("mktcap_bin"))

# df = df.to_dummies(columns=["mktcap_bin"])

# df = df.drop("mktcap_bin_2")


# print(df)



T = 60
days= 21*T
df = df.sort(["barrid","yyyymm"])

# # current period
df = df.with_columns([
    # MOMENTUM
    pl.col("specific_return").log1p().shift(42).rolling_sum(window_size=12*21).over("barrid").alias("curmom"),
    
    # volitility
    pl.col("specific_return").shift(0).rolling_std(window_size=days).over("barrid").alias("curvol"),
    
    # Skewness
    pl.col("specific_return").shift(0).rolling_skew(window_size=days).over("barrid").alias("curskew"),
    
    # Turnover
    # pl.col("turnover").shift(1).rolling_mean(window_size=T).over("barrid").alias("curturn")])
    pl.col("turnover").shift(21).rolling_sum(window_size=days).over("barrid").alias("curturn")])



    # mktbin
# # previous
df = df.with_columns([
#     # MOMENTUM
    pl.col("specific_return").log1p().shift(days+42).rolling_sum(window_size=12*21).over("barrid").alias("prevmom"),
    
    # volitility
    pl.col("specific_return").shift(days+21).rolling_std(window_size=days).over("barrid").alias("prevvol"),
    
    # Skewness
    pl.col("specific_return").shift(days).rolling_skew(window_size=days).over("barrid").alias("prevskew"),
    
    # Turnover
    # pl.col("turnover").shift(1+T).rolling_mean(window_size=T).over("barrid").alias("prevturn"),
    pl.col("turnover").shift(days+21).rolling_sum(window_size=days).over("barrid").alias("prevturn"),

    
    # mktcap
    pl.col("curmktb1").shift(21).over("barrid").alias("prevmktb1"),
    pl.col("curmktb2").shift(21).over("barrid").alias("prevmktb2"),
    pl.col("curmktb3").shift(21).over("barrid").alias("prevmktb3")])

# # last trading day per month
monthly_df = df.group_by(["barrid", "yyyymm"]).agg([
    pl.col("specific_return").last().alias("specific_return"),
    pl.col("turnover").mean().alias("turnover"),
    pl.col("market_cap").last().alias("market_cap"),
    pl.col("price").last().alias("price"),
    pl.col("predicted_beta").last().alias("predicted_beta"),
    pl.col("specific_risk").last().alias("specific_risk"),
    pl.col("date").last().alias("date"),
    pl.col("return").last().alias("return"),
    
    pl.col("prevmktb2").last().alias("prevmktb2"),
    pl.col("prevmktb1").last().alias("prevmktb1"),
    pl.col("curmktb2").last().alias("curmktb2"),
    pl.col("curmktb1").last().alias("curmktb1"),
    pl.col("curmktb3").last().alias("curmktb3"),
    pl.col("prevmktb3").last().alias("prevmktb3"), 
    
    pl.col("prevmom").last().alias("prevmom"),
    pl.col("prevturn").last().alias("prevturn"),   
    pl.col("prevskew").last().alias("prevskew"),
    pl.col("prevvol").last().alias("prevvol"),   
    
    pl.col("curmom").last().alias("curmom"),
    pl.col("curturn").last().alias("curturn"),   
    pl.col("curskew").last().alias("curskew"),
    pl.col("curvol").last().alias("curvol"), 
    
])

monthly_df = monthly_df.with_columns(pl.col('market_cap').shift(1).alias('mktcap_lag'))



# # Drop rows with missing values
monthly_df = monthly_df.drop_nulls(["curmom", "curvol", "curskew", "curturn", "market_cap", "prevmom", "prevvol", "prevskew", "prevturn",'prevmktb1', 'prevmktb2'])

# # ols per mounth
def cross_sectional_ols(df_month):
    X = df_month[["prevmom", "prevvol", "prevskew", "prevturn", 'prevmktb1','prevmktb2']]
    y = df_month["curskew"]  
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    coef_dict = model.params.to_dict()
    return model

# # Shift skew by -1 month to get next month skew
# monthly_df = monthly_df.sort(["barrid","yyyymm"])
# monthly_df = monthly_df.with_columns(
#     pl.col("curskew").shift(-1).over("barrid").alias("skew_next")  
# )

# monthly_df = monthly_df.drop_nulls(["skew_next"])

# # run cross-sectional regression
coeffs_list = []
monthly= []
monthly_df = monthly_df.sort(["barrid","yyyymm"])


for month, df_month in monthly_df.group_by("yyyymm"):
    df_month_pd = df_month.to_pandas()
    model = cross_sectional_ols(df_month_pd)
    coeffs_list.append(model)
    params = model.params.to_dict()

    monthly.append([month[0], params['const'],params['prevmom'], params['prevvol'], params['prevskew'], params['prevturn'], params['prevmktb1'], params['prevmktb2']])

# # monthly1 has all of the monthly results of the regression, merges it onto monthly_df
monthly1 = pd.DataFrame(monthly, columns=['yyyymm', 'const', 'Bprevmom', 'Bprevvol', 'Bprevskew', 'Bprevturn', 'Bprevmktb1', 'Bprevmktb2'])
monthly1 = pl.from_pandas(monthly1)
monthly_df= monthly_df.join(monthly1, on='yyyymm', how='left')


avgcoeffs = monthly1[['const', 'Bprevmom', 'Bprevvol', 'Bprevskew', 'Bprevturn', 'Bprevmktb1', 'Bprevmktb2']].mean()
print(avgcoeffs)

# # Compute expected skewness using past characteristics
monthly_df = monthly_df.with_columns(
    (
        pl.col("const")
        + pl.col("Bprevskew") * pl.col("curskew")
        + pl.col("Bprevvol") * pl.col("curvol")
        + pl.col("Bprevmom") * pl.col("curmom")
        + pl.col("Bprevturn") * pl.col("curturn")
        + pl.col("Bprevmktb1") * pl.col("curmktb1")
        + pl.col("Bprevmktb2") * pl.col('curmktb2')
    ).alias(signal_name)
)

# is this part right
# monthly_df= monthly_df.with_columns(pl.col(signal_name).shift(1).alias(signal_name))


# print('check5')



# Filter universe
filtered = monthly_df.filter(
    pl.col("price").shift(1).over("barrid").gt(price_filter),
    pl.col(signal_name).is_not_null(),
    pl.col("predicted_beta").is_not_null(),
    pl.col("specific_risk").is_not_null())



# Compute scores
scores = filtered.select(
    "date",
    "barrid",
    "predicted_beta",
    "specific_risk",
    "return",
    
    pl.col(signal_name)
    .sub(pl.col(signal_name).mean())
    .truediv(pl.col(signal_name).std())
    .over("date")
    .alias("score"),
)

# Compute alphas
alphas = (
    scores.with_columns(pl.col("score").mul(IC).mul(pl.col("specific_risk")).alias("alpha"))
    .select("date", "barrid", "alpha", "predicted_beta", "return")
    .sort("date", "barrid")
)
print(alphas)
alphas = alphas.drop_nulls()
alphas.write_parquet(f"{signal_name}_alphas.parquet")
