#!/usr/bin/env python
# coding: utf-8

# In[154]:


import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# In[155]:


# Stocks Consdered
stocks = ["ACN","^GSPC"]

#Start Date and End Date
start_date = "2017-01-01"
end_date = "2023-01-01"

#Risk Free Rate
rf = 0.05

#Months in year - to calculate annual return
m = 12


# In[156]:


#Downloading Data

def download_data(stocks, start_date, end_date):
    data = {}
    
    for stock in stocks:
        historical_data = yf.download(stock, start=start_date, end=end_date)
        data[stock] = historical_data['Adj Close']
    
    return pd.DataFrame(data)

df = download_data(stocks, start_date, end_date).resample('M').last()  #Changing data from daily to monthly
print(df)


# In[157]:


def ln_returns(df):
    log_return = np.log(df / df.shift(1))
    return log_return[1:]

ln_monthly_returns = ln_returns(df)
print(ln_monthly_returns)

matrix = pd.DataFrame({'stock_value': ln_monthly_returns[stocks[0]],'market_value': ln_monthly_returns[stocks[1]]})
print(matrix)


# In[158]:


def calculate_beta(matrix):
        # covariance matrix: the diagonal items are the variances
        # the matrix is symmetric: cov[0,1] = cov[1,0] !!!
        covariance_matrix = np.cov(matrix["stock_value"], matrix["market_value"])
        # calculating beta according to the formula
        beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
        print("Beta from formula: ", beta)
        
calculate_beta(matrix)


# In[159]:


# Calculating Regression

#Fitting linear regression line for data set

def regression(matrix):
    beta, alpha = np.polyfit(matrix['stock_value'], matrix['market_value'], deg = 1) #we are using deg=1 as we are fitting linear function, if deg=2 it will be quadratic function and drg=3 is cubic function
    print("Beta value from Regression: ", beta)
    
    
    # calculate the expected return according to the CAPM formula
    # we are after annual return (this is why multiply by 12)
    expected_return = rf + beta * ((matrix['market_value'].mean()*m) - rf)
    print("Expected Return: ", expected_return)
    return (alpha, beta)

regression(matrix)


# In[160]:


#Plotting

def plot_regression(matrix, alpha, beta):
    fig, axis = plt.subplots(1, figsize=(20, 10))
    axis.scatter(matrix["market_value"], matrix['stock_value'],
                     label="Data Points")
    axis.plot(matrix["market_value"], beta * matrix["market_value"] + alpha,
                  color='red', label="CAPM Line")
    plt.title('Capital Asset Pricing Model, finding alpha and beta')
    plt.xlabel('Market return $R_m$', fontsize=18)
    plt.ylabel('Stock return $R_a$')
    plt.text(0.08, 0.05, r'$R_a = \beta * R_m + \alpha$', fontsize=18)
    plt.legend()
    plt.grid(True)
    plt.show()
alpha, beta = regression(matrix)  # Get both alpha and beta
plot_regression(matrix, alpha, beta)  

