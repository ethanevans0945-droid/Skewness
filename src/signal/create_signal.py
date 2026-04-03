import polars as pl
import datetime as dt
import sf_quant.data as sfd
import polars_ols


start = dt.date(2000, 1, 1)
end = dt.date(2024, 12, 31)
signal_name = "IS"
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
days=21*T

#calculate turnover
df = data.sort(["barrid", "date"])
df = df.with_columns(
    (pl.col("daily_volume") / pl.col("market_cap")).alias("turnover")
)

df = df.sort(["barrid","date"]).with_columns(pl.col('turnover').shift(days).rolling_mean(window_size=T).over('barrid').alias(f'per1turn'))
df = df.sort(["barrid","date"]).with_columns(pl.col('turnover').rolling_mean(window_size=T).over('barrid').alias(f'per2turn'))


#sort
df = df.sort(["barrid", "date"])

# finds momentum 
df = df.with_columns(pl.col('specific_return').log1p().shift(days).rolling_sum(window_size=days+(12*21)).over('barrid').alias('per1mom'))
df = df.sort(["barrid", "date"])
df = df.with_columns(pl.col('specific_return').log1p().rolling_sum(window_size=days+(12*21)).over('barrid').alias('per2mom'))
df = df.sort(["barrid", "date"])

# finds volitility
df = df.with_columns((pl.col('specific_return')).shift(days).rolling_std(window_size=days).over('barrid').alias('per1vol'))
df = df.sort(["barrid", "date"])
df = df.with_columns((pl.col('specific_return')).rolling_std(window_size=days).over('barrid').alias('per2vol'))
df = df.sort(["barrid", "date"])

# finds skewness
df = df.with_columns(pl.col('specific_return').shift(days).rolling_skew(window_size=days).over('barrid').alias('per1skew'))
df = df.sort(["barrid", "date"])

df = df.with_columns(pl.col('specific_return').rolling_skew(window_size=days).over('barrid').alias('per2skew'))
df = df.sort(["barrid", "date"])


df = df.drop_nulls(["per1vol", "per1skew", "per2skew", "per1mom", "per1turn"])

result = (
    df.group_by("barrid")
    .agg(
        pl.col("per2skew")
        .least_squares.ols(
            pl.col(["per1vol","per1skew", "per1mom", "per1turn"]),
            mode="coefficients",
            add_intercept=True
        ).alias("coefficients")))
    
result_expanded = (result.unnest("coefficients").rename({
        "per1turn": "Bturn",
        "per1skew": "Bskew",
        "per1vol": "Bvol",
        "per1mom": "Bmom",
        "const": "Intercept"}))
    # use betas to predict futurn skewness 

df = df.join(result_expanded, on='barrid', how='left')
df = df.with_columns((pl.col('Intercept') + pl.col('per2skew')*pl.col('Bskew') + pl.col('per2vol')*pl.col('Bvol') + pl.col('Bmom')*pl.col('per2mom') + pl.col('Bturn')*pl.col('per2turn')).alias(signal_name))
    





# Filter universe
filtered = df.filter(
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