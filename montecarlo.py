import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

#plt.style.use('seaborn-vintage')
sns.set_context('talk')

ticker = 'AMZN]'
start_date = '2020-01-01'
end_date = '2024-12-31'

data = yf.download(ticker, start=start_date, end=end_date)
data = data['Close'].squeeze()
data.dropna(inplace=True)

# plt.figure(figsize=(12,5))
# plt.plot(data, label=f'{ticker} Adjusted Close')
# plt.title(f'{ticker} Price History')
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.legend()
# plt.show()

log_returns = np.log(data / data.shift(1)).dropna()

mu = float(log_returns.mean())
sigma = float(log_returns.std())

print(f"Daily Mean Return: {mu:.5f}")
print(f"Daily Volatility {sigma:.5f}")
print(f"Annulaized Return: {mu * 252:.3f}")
print(f"Annulaized Volatility: {sigma * np.sqrt(252):.3f}")


num_simulations = 1000
num_days = 252
last_price = data.iloc[-1]

simulated_prices = np.zeros((num_days, num_simulations))

for sim in range(num_simulations):
    prices = [last_price]
    for day in range(1, num_days):
        random_shock = np.random.normal(0,1)
        price = prices[-1] * np.exp((mu - 0.5 * sigma**2) +sigma *random_shock)
        prices.append(price)
    simulated_prices[:, sim] = prices

print(f"Simulated Prices: {simulated_prices}")

plt.figure(figsize=(14,6))
plt.plot(simulated_prices, alpha=0.1, color='blue')
plt.title(f'{num_simulations} Simulated Price Paths for {ticker}')
plt.xlabel('Days')
plt.ylabel('Price')
plt.grid(True)
plt.show()

ending_prices = simulated_prices[-1, :]
mean_price = np.mean(ending_prices)
var_95 = np.percentile(ending_prices, 5)

print(f"\nExpected Final Price (Mean): ${mean_price:.2f}")
print(f"%5 Quantile (Value at Risk): ${var_95:.2f}")
print(f"Chance of Loss: {np.mean(ending_prices < last_price) * 100:.2f}%")
plt.figure(figsize=(12,6))
sns.histplot(ending_prices, bins=100, kde=True, color='skyblue')
plt.axvline(last_price, color='red', linestyle='--', label = 'Current Price')
plt.axvline(mean_price, color='green', linestyle='--', label='Mean Final Price')
plt.axvline(var_95, color='orange', linestyle='--', label='5% VaR')
plt.title(f'Distrubtion of Simulated Final Prices for {ticker}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Monte Carlo Stock Price Simulation
# This Python script implements a Monte Carlo simulation to forecast future stock price movements using geometric Brownian motion. 
# The model downloads historical stock data for a company from 2020-2024, 
# calculates key statistical parameters (daily mean return and volatility), 
# and uses these to simulate 1,000 possible price paths over the next 252 trading days (one year). 
# Each simulation applies random shocks from a normal distribution to model the unpredictable nature 
# of market movements while incorporating the stock's historical drift and volatility patterns.
# The simulation engine runs 1,000 independent forecasts, 
# where each day's price is calculated using the geometric Brownian motion formula: 
# S(t+1) = S(t) × e^[(μ - 0.5σ²) + σ×Z]. 
# This mathematical model captures both the expected return component (drift) 
# and the random volatility component that characterizes real stock price behavior. 
# The code tracks all simulated price paths and stores the final prices from each simulation 
# to analyze the distribution of possible outcomes after one year.
# The results provide valuable risk management insights including the expected final stock price, 
# the 5% Value at Risk (VaR) representing the worst-case scenario for 95% confidence, 
# and the probability of experiencing a loss. Two visualizations are generated: a line chart 
# showing all 1,000 simulated price trajectories, and a histogram displaying the distribution of final prices 
# with reference lines for current price, mean expected price, and VaR threshold. This approach helps investors 
# understand the range of possible outcomes and make more informed decisions based on probabilistic rather than 
# deterministic forecasts.