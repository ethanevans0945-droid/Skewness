import os
import polars as pl
import datetime as dt
from dotenv import load_dotenv
import sf_quant.data as sfd
import polars_ols



def load_data() -> pl.DataFrame:
    """
    Load and prepare market data for signal creation.

    Returns:
        pl.DataFrame: Market data with required columns
    """
    # TODO: Load data from source (API, file, database)

    # TODO: Filter data as needed (date range, symbols, quality checks)
    start = dt.date(1929, 1, 1)
    end = dt.date(2024, 12, 31)

    # columns = [
    #     'date',
    #     'barrid',
    #     'ticker',
    #     'price',
    #     'return',
    #     'specific_return',
    #     'predicted_beta'
    # ]
    
    columns = [
        'date',
        'barrid', 'price', 'ticker', 'return', 'specific_return', 'specific_risk', 'predicted_beta'
    ]
    
    df = sfd.load_assets(
        start=start,
        end=end,
        in_universe=True,
        columns=columns)
        
    df = df.with_columns(pl.col('date').dt.truncate('1mo').alias('month'))
    # df= df.sort(['barrid', 'month']).with_columns(pl.col('month').shift(1).over('barrid').alias(f'monthlyret'))
    df = df.sort(["barrid", "month"])
    df = df.with_columns(
    pl.col('return')
    .first()
    .over(['barrid', 'month'])
    .alias('monthreturn')
)
    return df
    

# data=load_data()
# print(data)

def create_signal():
    """
    Loads data, creates a simple signal, and saves it to parquet.
    """
    # Load environment variables from .env file
    load_dotenv()
    project_root = os.getcwd()
    output_path = os.getenv("SIGNAL_PATH", "data/signal.parquet")
    if not os.path.isabs(output_path):
        output_path = os.path.join(project_root, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)


    # TODO: Load Data
    df = load_data()
    
    # TODO: Add your signal logic here (remember alpha logic)

    T=60
    days= T*21
    
    
    
    

    df = df.with_columns((pl.col('specific_return').truediv(100)).alias('specific_return')) #return space
    df = df.with_columns((pl.col('return').truediv(100)).alias('return')) #return space
    df = df.with_columns((pl.col('monthreturn').truediv(100)).alias('monthreturn'))


    df = df.with_columns(pl.col('specific_return').rolling_mean(window_size=days).over('barrid').alias('mu'))
    df = df.with_columns((pl.col('specific_return')).rolling_std(window_size=days).over('barrid').alias('iv')) 
    df = df.with_columns((((pl.col('specific_return') - pl.col('mu'))**3/pl.col('iv')**3)).over('barrid').alias('m3'))
    df = df.with_columns(pl.col('m3').rolling_sum(window_size=days).truediv(days/((days-1)*(days-2))).alias('is'))
    
   
    
    # forward fills from daily to monthly
    df = df.with_columns(pl.col("date").dt.truncate("1mo").alias("month"))

    df = df.with_columns(pl.col("is").first().over("barrid","month").alias("ismonth"))
    df = df.drop_nulls(["is", "iv", "ismonth"])
    df = df.drop(['m3', 'mu'])
    

    
    
    # alpha logic
    df=(df.sort('barrid', 'date')
        .with_columns(
            pl.col('specific_risk').fill_null(strategy='forward').over('barrid')
        )
        .with_columns(
            pl.col('is')
            .sub(pl.col('is').mean())
            .truediv(pl.col('is').std())
            .over('date')
            .alias('score')).with_columns(
            pl.lit(0.05).mul('score').mul('specific_risk').alias('alpha')))
    

    # lags price and filters
    df= df.sort('barrid', 'date').with_columns(
            pl.col('price').shift(1).over('barrid').alias('price_lag')).filter(
            pl.col('price_lag').gt(5),
            pl.col('alpha').is_not_null()).sort('barrid', 'date')
    
    
    # return df
    
    
    
    # TODO: Save to data/signal.parquet

    df.write_parquet(output_path)

if __name__ == "__main__":
    create_signal()
