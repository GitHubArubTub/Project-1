import numpy as np
import pandas as pd

class MCSimulation:
    def __init__(self, portfolio_data, weights=None, num_simulation=1000, num_trading_days=252):
        if not isinstance(portfolio_data, pd.DataFrame):
            raise TypeError("portfolio_data must be a Pandas DataFrame")

        num_stocks = len(portfolio_data.columns.get_level_values(0).unique())
        self.weights = np.array(weights if weights else [1.0/num_stocks]*num_stocks)
        self.weights /= self.weights.sum()

        if "daily_return" not in portfolio_data.columns.get_level_values(1).unique():
            close_df = portfolio_data.xs('close',level=1,axis=1).pct_change()
            tickers = portfolio_data.columns.get_level_values(0).unique()
            column_names = [(x,"daily_return") for x in tickers]
            close_df.columns = pd.MultiIndex.from_tuples(column_names)
            portfolio_data = portfolio_data.merge(close_df,left_index=True,right_index=True).reindex(columns=tickers,level=0)    

        self.portfolio_data = portfolio_data
        self.nSim = num_simulation
        self.nTrading = num_trading_days
        self.simulated_return = ""

    def calc_cumulative_return(self):
        last_prices = self.portfolio_data.xs('close',level=1,axis=1)[-1:].values.tolist()[0]
        daily_returns = self.portfolio_data.xs('daily_return',level=1,axis=1)
        mean_returns = daily_returns.mean().tolist()
        std_returns = daily_returns.std().tolist()

        results = []

        for n in range(self.nSim):
            if n % 10 == 0:  # print progress every 10 simulations
                print(f"Running Monte Carlo simulation number {n}.")
            simvals = [[p] for p in last_prices]
            for s in range(len(last_prices)):
                for i in range(self.nTrading):
                    simvals[s].append(simvals[s][-1] * (1 + np.random.normal(mean_returns[s], std_returns[s])))
            sim_df = pd.DataFrame(simvals).T.pct_change()
            sim_df = sim_df.dot(self.weights)
            results.append((1 + sim_df.fillna(0)).cumprod())

        portfolio_cumulative_returns = pd.concat(results, axis=1)
        self.simulated_return = portfolio_cumulative_returns

        self.confidence_interval = portfolio_cumulative_returns.iloc[-1, :].quantile(q=[0.025, 0.975])
        return portfolio_cumulative_returns



    def plot_simulation(self):
        plot_title = f"{self.nSim} Simulations of Cumulative Portfolio Return Trajectories Over the Next {self.nTrading} Trading Days."
        return self.simulated_return.plot(legend=None,title=plot_title)
    
    def plot_distribution(self):
        plot_title = f"Distribution of Final Cumuluative Returns Across All {self.nSim} Simulations"
        plt = self.simulated_return.iloc[-1, :].plot(kind='hist', bins=10,density=True,title=plot_title)
        plt.axvline(self.confidence_interval.iloc[0], color='r')
        plt.axvline(self.confidence_interval.iloc[1], color='r')
        return plt
    
    def summarize_cumulative_return(self):
        # Check to make sure that simulation has run previously. 
        if not isinstance(self.simulated_return,pd.DataFrame):
            self.calc_cumulative_return()
        metrics = self.simulated_return.iloc[-1].describe()
        ci_series = self.confidence_interval
        ci_series.index = ["95% CI Lower","95% CI Upper"]
        return pd.concat([metrics, ci_series])
