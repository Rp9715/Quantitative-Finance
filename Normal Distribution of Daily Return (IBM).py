#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# In[11]:


# Stocks Consdered
stocks = ["IBM"]

#Start Date and End Date
start_date = "2012-01-01"
end_date = "2022-01-01"


# In[12]:


#Gathering Data

def import_data():
    
    stock_data = {}
    
    for stock in stocks:  #closing value
        ticker = yf.Ticker(stock)
        data = ticker.history(start = start_date, end = end_date)["Close"]
        data.index = data.index.date   # removing time from data
        data = data.round(3)    #Rounding off data upto 3 decimal place
        stock_data[stock] = data
        
    return pd.DataFrame(stock_data)

print(import_data())


# In[13]:


# Calculating Returns

def ln_returns(data):  # we calculate log returns so that it can be normalized and we can measure all variables in comparable matric
    log_return = np.log(data/data.shift(1))
    return log_return[1:]

log_daily_returns = ln_returns(import_data()) 
print(log_daily_returns)


# In[19]:


def show_plot(log_daily_returns):
    plt.hist(log_daily_returns, bins=300)
    stock_variance = log_daily_returns.var()
    stock_mean = log_daily_returns.mean()
    sigma = np.sqrt(stock_variance)
    x = np.linspace(stock_mean - 3 * sigma, stock_mean + 3 * sigma, 100)
    plt.title('Normal Distribution of IBM Stock')
    plt.show()
show_plot(log_daily_returns)


# In[ ]:




