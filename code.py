from backtesting import Backtest,Strategy
import yfinance as yf
from backtesting.lib import crossover
import pandas as pd
import numpy as np
import talib
from backtesting.test import GOOG
import matplotlib.pyplot as plt


# Here i download data of indian stokes
df = yf.download('jmfinancil.NS', start='2023-01-03', end='2025-5-20', auto_adjust=False)
if df.empty:
    print("No data available. Exiting.")
    exit()
df.columns = df.columns.get_level_values(0)
df.dropna(inplace=True)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df.index.name = 'Date'

#Here i define a function name TR which here i used to estimate the volatility
# it basicllay calculate the average of lenth of (max-low (price))abs of 14 day period, means show normal movement of a price
def TR(df,period=14):
    high=df.High
    low=df.Low
    close = df.Close
    # true range
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    return atr

# measure upward direction movement,indicate buying pressure
def DM(df,period=14):
    high=df.High
    low=df.Low
    up_move = high - high.shift()
    down_move = low.shift() - low
    # +DM: If up_move > down_move and up_move > 0 → up_move, else 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    return pd.Series(plus_dm, index=df.index).rolling(period).mean()

# measure downward direction movement,indicate selling pressure
def DM1(df,period=14):
    high=df.High
    low=df.Low
    up_move = high - high.shift()
    down_move = low.shift() - low
    # -DM: If down_move > up_move and down_move > 0 → down_move, else 0
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    return pd.Series(minus_dm, index=df.index).rolling(period).mean()

# here i define a class name my_stratergy , in which i define my stratergy
class my_stratergy(Strategy):

    tr_period=14

    entry_z = 1.5
    exit_z = 0.5
    z_window=20

    window_long_sma1=90
    window_short_sma=30

# mcad data
    ema_fast=12 #period of fast ema
    ema_slow=26 #period of slow ema
    ema_mid=9  #period of mid ema
# rsi data
    rsi_window=14  #period of rsi
    upper_bound_RSI =70  # upper bound of rsi
    lower_bound_RSI =30  # lower bound of rsi
    window_long_sma1=90
    window_short_sma=30
    
# bollinger  band data
    price_window=20

# function in which indicator or other useful thing for stratergy is define . In this the things are run for all the data (means for every point on chart)    
    def init(self): 
       price=pd.Series(self.data.Close)
       # mcad (trend strength)
       self.ema_fast=self.I(lambda x: pd.Series(x).ewm(span=self.ema_fast,adjust=False).mean(),self.data.Close) # ema for close price data of 12 days
       self.ema_slow=self.I(lambda x: pd.Series(x).ewm(span=self.ema_slow,adjust=False).mean(),self.data.Close)  # ema for close price data of 26 days  
       self.mcad=self.I(lambda x,y: x-y,self.ema_fast,self.ema_slow)   # ema(12)-ema(26)
       self.ema_mid=self.I(lambda x: pd.Series(x).ewm(span=self.ema_mid,adjust=False).mean(),self.mcad)  #  ema for close price data of 9 days
#  rsi (momentum oscillator)

       self.rsi=self.I(talib.RSI,self.data.Close,self.rsi_window) # here i define a rsi indicator
       self.long_sma1=self.I(talib.SMA,self.data.Close,self.window_long_sma1)  #LONG SMA of 90 days
       self.short_sma=self.I(talib.SMA,self.data.Close,self.window_short_sma)   # SHORT SMA for 30 days
# adx ( trend strength)
       self.atr=self.I(TR,self.data.df,self.tr_period)  
       self.dm=self.I(DM,self.data.df,self.tr_period) # positive direction movement
       self.dm1=self.I(DM1,self.data.df,self.tr_period)# negative direction movement
       self.DIP=self.I(lambda x,y: (x/y)*100 ,self.dm,self.tr_period) # Higher DIP means buyers are stronger
       self.DIN=self.I(lambda x,y: (x/y)*100 ,self.dm1,self.tr_period) #Higher +DIN means SELLERS are stronger
       self.DX=self.I(lambda x,y: abs((((x-y)/(x+y))*100)),self.DIP,self.DIN) # IT basically show dominance of DIP ,DIN which give signal of a strong trend if DX is higher
       self.ADX=self.I(lambda x: pd.Series(x).rolling(self.tr_period).mean(),self.DX) # it is average of DX over a  period of 14 days
# z score ()
       self.mean=self.I(lambda x: x.rolling(self.z_window).mean(),price) # moving average of price(close data) of 20 days
       self.std=self.I(lambda x: x.rolling(self.z_window).std(),price)   # std of price(close data) for 20 days
    #    self.zscore=self.I(lambda x,m,s: (x-m)/s,price,self.mean,self.std)  # z_score=(price-mean)/std

 #  bollinger band ( based on volatility, reversal )
       price=pd.Series(self.data.Close)
       self.middle=self.I(lambda x: x.rolling(self.price_window).mean(),price)     # moving average of price(close data) of 20 days
       self.upper=self.I(lambda x,std: x.rolling(self.price_window).mean()+2*std,price,self.std)  # moving average + 2*std(close data price ,20 days)
       self.lower=self.I(lambda x,std: x.rolling(self.price_window).mean()-2*std,price,self.std)  # moving average - 2*std(close data price ,20 days)

      
    def next(self):
      price=self.data.Close[-1]
      ADX=self.ADX[-1]
            
      if (ADX>25)   :
          if((crossover(self.DIP, self.DIN))):
             if(not(self.position)):
              self.buy()
             elif((self.position)):  
              self.position.close()
              self.buy()

          elif (crossover(self.DIN, self.DIP)):
               if(not(self.position)):
                 tp=price*0.9
                 self.sell(tp=tp)
               elif((self.position)): 
                self.position.close()
                self.sell()
      elif(ADX<20):
        if (self.position):
              self.position.close()

 
      if crossover(self.upper,price)|crossover (self.rsi,self.upper_bound_RSI):
          self.sell()          
      elif crossover(self.lower_bound_RSI,self.rsi) or((crossover(self.short_sma,self.long_sma1)&(self.rsi<self.upper_bound_RSI))|crossover(price,self.lower)):
            sl=price*0.95
            size=.8
            self.buy(size=size,sl=sl)      
      elif (crossover(self.long_sma1,self.short_sma)&(self.rsi<self.upper_bound_RSI)):
            self.position.close()
              

# here i run my stratergy              
bt=Backtest(df,my_stratergy,cash=10000000)    
stats2=bt.run()
bt.plot(filename="JMFINANCIL.html")
print(stats2)

equity = stats2['_equity_curve']
trades = stats2['_trades']

#Plot Equity Curve
plt.figure(figsize=(10, 4))
plt.plot(equity['Equity'])
plt.title('Equity Curve For JMFINANCIL')
plt.xlabel('Time')
plt.ylabel('Portfolio Value')
plt.grid()
plt.show()

#  Plot Trade Curve
cumulative_pnl = np.cumsum(trades['PnL'])
plt.figure(figsize=(10, 4))
plt.plot(cumulative_pnl)
plt.title('Trade Curve FOR JMFINANCIL')
plt.xlabel('Trade Number')
plt.ylabel('Cumulative Profit')
plt.grid()
plt.show()

#  Plot Drawdown Curve
equity['Peak'] = equity['Equity'].cummax()
equity['Drawdown'] = (equity['Equity'] - equity['Peak']) / equity['Peak']
plt.figure(figsize=(10, 4))
plt.plot(equity['Drawdown'])
plt.title('Drawdown Curve')
plt.xlabel('Time')
plt.ylabel('Drawdown')
plt.grid()
plt.show()

# Save trades to a CSV file
trades = stats2['_trades']
print(trades)
print(trades[['EntryTime', 'ExitTime', 'EntryPrice', 'ExitPrice', 'PnL', 'ReturnPct']])
trades.to_csv('coalindia_trades.csv', index=False)
