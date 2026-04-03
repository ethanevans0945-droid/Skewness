from sf_backtester import BacktestConfig, BacktestRunner, SlurmConfig

slurm_config = SlurmConfig(
    n_cpus=8,
    mem="32G",
    time="03:00:00",
    mail_type="BEGIN,END,FAIL",
    max_concurrent_jobs=30,
)

backtest_config = BacktestConfig(
    signal_name="IS",
    data_path="/home/evans945/Projects/Skewness/src/signal/IS_alphas.parquet",
    gamma=50,
    project_root="/home/evans945/Projects/Skewness/src/signal/",
    byu_email="evans945@byu.edu",
    constraints=["ZeroBeta", "ZeroInvestment"],
    slurm=slurm_config,
)

backtest_runner = BacktestRunner(backtest_config)
backtest_runner.submit(dry_run=False)