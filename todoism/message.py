def allocate(
    invest_date="2022-01-30",
    idx="DOW", 
    sp500_path="tickers.csv", 
    ddpg_path="trained_ddpg.zip",
    sac_path="trained.zip",
    money=1000000,
    buy_cost_pct=0.001, 
    sell_cost_pct=0.001
    ):
    """
    money: int
    idx: "DOW" or "SP500"
    actions: dict {ticker:number}
    """

    def get_env(
        trade_data, 
        money, 
        buy_cost_pct, 
        sell_cost_pct
        ):
        """
        trade_data: df
        """

        from finrl.finrl_meta.env_stock_trading.env_stocktrading import StockTradingEnv

        stock_dimension = len(trade_data.tic.unique())
        state_space = 1 + 2 * stock_dimension + \
            len(config.TECHNICAL_INDICATORS_LIST) * stock_dimension

        print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

        env_kwargs = {
            "hmax": 100, 
            "initial_amount": money,
            "buy_cost_pct": buy_cost_pct,
            "sell_cost_pct": sell_cost_pct,
            "state_space": state_space, 
            "stock_dim": stock_dimension, 
            "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST, 
            "action_space": stock_dimension, 
            "reward_scaling": 1e-4
        }

        trade_env = StockTradingEnv(
            df = trade_data, 
            turbulence_threshold = 70, 
            risk_indicator_col='vix', 
            **env_kwargs
            )

        return trade_env

    def download(invest_date, idx, sp500_path):

        # def get_sp500(path):
        #     import pandas as pd
        #     return pd.read_csv(path)['Symbol'].values.tolist()
        # from finrl.finrl_meta.preprocessor.yahoodownloader import YahooDownloader
        from datetime import date

        today = date.today().strftime("%Y-%m-%d")
        # tickers = get_sp500(sp500_path) if idx == "SP500" else config.DOW_30_TICKER
        # df = YahooDownloader(
        #     start_date = invest_date,
        #     end_date = today,
        #     ticker_list = tickers
        #     ).fetch_data().sort_values(['date','tic'],ignore_index=True)

        df = pd.read_csv("fin_data.csv")

        return df[(df['date'] >= invest_date) & (df['date'] <= today)]

    def preprocess(df, invest_date):
        """
        df: dataframe
        """

        from finrl.finrl_meta.preprocessor.preprocessors import FeatureEngineer, data_split
        from finrl.finrl_meta.data_processor import DataProcessor
        import itertools
        from datetime import date

        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list = config.TECHNICAL_INDICATORS_LIST,
            use_vix=True,
            use_turbulence=True,
            user_defined_feature=False
            )
        
        processed = fe.preprocess_data(df)
        
        list_ticker = processed["tic"].unique().tolist()
        list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
        combination = list(itertools.product(list_date,list_ticker))

        processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
        processed_full = processed_full[processed_full['date'].isin(processed['date'])]
        processed_full = processed_full.sort_values(['date','tic'])

        processed_full = processed_full.fillna(0)
        today = date.today().strftime("%Y-%m-%d")
        # print("Processed shape: ", processed_full.shape)
        # print("Today: ", today)
        # print("Invest date: ", invest_date)
        trade_data = data_split(processed_full, invest_date, today)
        print(trade_data.shape)
        
        # Set turbulence threshold
        data_risk_indicator = processed_full[(processed_full.date <= today) & (processed_full.date >= invest_date)]
        insample_risk_indicator = data_risk_indicator.drop_duplicates(subset=['date'])

        return trade_data, insample_risk_indicator
    
    import pandas as pd
    # import numpy as np
    # import matplotlib
    # import matplotlib.pyplot as plt

    # %matplotlib inline
    from finrl.apps import config
    from finrl.drl_agents.stablebaselines3.models import DRLAgent, DDPG, SAC
    # from finrl.finrl_meta.data_processor import DataProcessor
    # from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
    # from pprint import pprint

    import sys
    sys.path.append("../FinRL-Library")

    # Download data from YahooFinance
    df = download(invest_date, idx, sp500_path)
    
    # Pre-Process
    trade_data, _ = preprocess(df, invest_date)

    # Env
    trade_env = get_env(
        trade_data,
        money, 
        buy_cost_pct, 
        sell_cost_pct
        )

    # Trade
    trained = SAC.load("trained")
    portfolio_value, actions = DRLAgent.DRL_prediction(
        model=trained,
        environment=trade_env
        )

    return portfolio_value, actions