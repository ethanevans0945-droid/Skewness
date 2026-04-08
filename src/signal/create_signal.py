import polars as pl
import datetime as dt
import sf_quant.data as sfd
import polars_ols
import statsmodels.api as sm
import pandas as pd



start = dt.date(2000, 1, 1)
end = dt.date(2024, 12, 31)
signal_name = "IS2"
price_filter = 5
IC = 0.05

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

T=5
days=T*21

df = data.sort(["barrid", "date"])


# calculates turnover 
df = df.with_columns(
    (pl.col("daily_volume") / pl.col("market_cap")).alias("turnover")
)

# Get last trading day of each month per stock
df = df.with_columns([
    pl.col("date").cast(pl.Datetime).alias("date_dt"),
    pl.col("market_cap").log().alias("log_mktcap")
])

# Create month indicator
df = df.with_columns(
    (pl.col("date_dt").dt.year() * 100 + pl.col("date_dt").dt.month()).alias("yyyymm")
)
print("check1")

# last trading day per month
monthly_df = df.group_by(["barrid", "yyyymm"]).agg([
    pl.col("specific_return").last().alias("specific_return"),
    pl.col("turnover").mean().alias("turnover"),
    pl.col("log_mktcap").last().alias("log_mktcap"),
    pl.col("price").last().alias("price"),
    pl.col("predicted_beta").last().alias("predicted_beta"),
    pl.col("specific_risk").last().alias("specific_risk"),
    pl.col("date").last().alias("date"),
    pl.col("return").last().alias("return")
])

# We'll shift by 1 month to skip last month
T = 12 
monthly_df = monthly_df.sort(["barrid","yyyymm"])

monthly_df = monthly_df.with_columns([
    # MOMENTUM
    pl.col("specific_return").log1p().shift(1).rolling_sum(window_size=T).over("barrid").alias("mom_12m"),
    
    # volitility
    pl.col("specific_return").shift(1).rolling_std(window_size=T).over("barrid").alias("vol_12m"),
    
    # Skewness
    pl.col("specific_return").shift(1).rolling_skew(window_size=T).over("barrid").alias("skew_12m"),
    
    # Turnover
    pl.col("turnover").shift(1).rolling_mean(window_size=T).over("barrid").alias("turn_12m")
])
print('check2')

# Drop rows with missing values
monthly_df = monthly_df.drop_nulls(["mom_12m", "vol_12m", "skew_12m", "turn_12m", "log_mktcap"])
print('check3')
# ols per mounth
def cross_sectional_ols(df_month):
    X = df_month[["skew_12m","vol_12m","mom_12m","turn_12m","log_mktcap"]]
    y = df_month["specific_return"]  
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model.params

# Shift skew by -1 month to get next month skew
monthly_df = monthly_df.sort(["barrid","yyyymm"])
monthly_df = monthly_df.with_columns(
    pl.col("skew_12m").shift(-1).over("barrid").alias("skew_next")  
)

monthly_df = monthly_df.drop_nulls(["skew_next"])

# run cross-sectional regression
coeffs_list = []

for month, df_month in monthly_df.group_by("yyyymm"):
    df_month_pd = df_month.to_pandas()  # statsmodels expects pandas
    params = cross_sectional_ols(df_month_pd)
    coeffs_list.append(params)

coeffs_df = pd.DataFrame(coeffs_list)
print('check4')
# average coefficients over months
avg_coeffs = coeffs_df.mean()

print(avg_coeffs)

# Compute expected skewness using past characteristics
monthly_df = monthly_df.with_columns(
    (
        avg_coeffs["const"]
        + avg_coeffs["skew_12m"] * pl.col("skew_12m")
        + avg_coeffs["vol_12m"] * pl.col("vol_12m")
        + avg_coeffs["mom_12m"] * pl.col("mom_12m")
        + avg_coeffs["turn_12m"] * pl.col("turn_12m")
        + avg_coeffs["log_mktcap"] * pl.col("log_mktcap")
    ).alias(signal_name)
)


print('check5')



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

alphas.write_parquet(f"{signal_name}_alphas.parquet")