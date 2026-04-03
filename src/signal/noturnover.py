import os
import polars as pl
import datetime as dt
from dotenv import load_dotenv
import sf_quant.data as sfd


def load_data() -> pl.DataFrame:
    """
    Load and prepare market data for signal creation.

    Returns:
        pl.DataFrame: Market data with required columns
    """
    # TODO: Load data from source (API, file, database)

    # TODO: Filter data as needed (date range, symbols, quality checks)
    data= sfd.get_assets_columns()
    
    print(data)
    start = dt.date(2024, 1, 1)
    end =  dt.date(2024, 12, 31)

    columns = [
        'date',
        'barrid', 'return', 'specific_return'
    ]


    # add filters
    df = sfd.load_assets(
        start=start,
        end=end,
        in_universe=True,
        columns=columns
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
    # update skewness calculation
    df = df.with_columns((pl.col('specific_return').truediv(100)).alias('specific_return')) #return space
    
    df = df.with_columns((pl.col('specific_return')).rolling_std(window_size=90).over('barrid').alias('rollstd')) #finds the idiosyncratic std
    df = df.with_columns((pl.col('specific_return')**3).rolling_mean(window_size=90).over('barrid').alias('num'))
    df = df.with_columns((pl.col('num'))/(pl.col('rollstd')**3).over('barrid').alias('skew'))
    

    # TODO: Save to data/signal.parquet

    df.write_parquet(output_path)

if __name__ == "__main__":
    create_signal()
