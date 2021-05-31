import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from pandas import DataFrame, Series
from pandas.io.parsers import TextFileReader
#Import data set -LOAN APPROVALS
loan_data = pd.read_csv("Loan_Train.csv")
#Take a look at the data
print(loan_data.info())
print(loan_data.head())
print(loan_data.shape)
# Count missing values in each column
missing_values_count = loan_data.isnull().sum()
print(missing_values_count[0:13])
#Replace .fillna() bfill or ffill or drop .dropna() missing values
clean_loandata=loan_data.fillna({'Credit_History':-1.0, 'Loan_Amount_Term':1000, 'LoanAmount':1000,
                                 'Self_Employed': 'Not Known', 'Dependents':-1, 'Married' : 'Not Known',
                                 'Gender':'Not Known'})
clean_loandata.info()
#Check details of columns
print(clean_loandata['Gender'].value_counts())
print(clean_loandata['Loan_Status'].value_counts())
print(clean_loandata['Property_Area'].value_counts())
print(clean_loandata['Education'].value_counts())
#Drop duplicates - no duplicates
drop_duplicates=clean_loandata.drop_duplicates()
drop_duplicates.info()
#Bar chart - loan amount by education
plt.figure(figsize=(8, 6))
sns.barplot(x='Education', y= 'LoanAmount',palette = 'GnBu', data=clean_loandata, capsize=0.05,
           saturation=5,errcolor='lightblue', errwidth=2)
plt.xlabel("Education")
plt.ylabel("Loan Amount")
plt.title("Loan Amount by Education")
plt.show()
#Bar chart - Applicant income by education
plt.figure(figsize=(8, 6))
sns.barplot(x='Education', y= 'ApplicantIncome',palette = 'GnBu', data=clean_loandata, capsize=0.05,
           saturation=5,errcolor='lightblue', errwidth=2)
plt.xlabel("Education")
plt.ylabel("Applicant Income")
plt.title("Applicant Income by Education")
plt.show()

#NEW DATASET
# Import a csv file into a Pandas dataframe S&P 500 Companies
companies= pd.read_csv("financials 2.csv")
# Take a look at the data
print(companies.columns)
print(companies.shape)
print(companies.info())
print(companies.isnull().sum())
#Price/Earnings missing 2
print(companies[companies['Price/Earnings'].isnull()])
#Price/Book is missing 8
print(companies[companies['Price/Book'].isnull()])
#Group by sector
sector=companies.groupby('Sector')
print(sector.first())
#Pick one sector from companies - Real Estate
realestate=companies[companies['Sector']=='Real Estate']
print(realestate.head(5))
realestatetop=realestate.sort_values(by='Earnings/Share', ascending=False).head(3)
print(realestatetop)
#Calculate no1 company, over all sectors, by P/S=P/E*E/S
companies['Price/Share']=companies['Price/Earnings']*companies['Earnings/Share']
print(companies.head(1))
#head.()
print(companies.head())
#Pie chart - sector by dividend yield in percentage
companies.groupby('Sector')['Dividend Yield'].sum().plot.pie(autopct="%.1f%%",figsize=(10,5))
plt.show()
#Bar chart - sector by price
fig=plt.figure(figsize=(20,5))
sns.barplot(x='Sector',y='Price',data=companies,palette='rainbow')
plt.ylabel("Price")
plt.xlabel("Sector")
plt.title("Sector by Price")
plt.tight_layout()
plt.show()
#Plot a bar chart - companies by count
ax = sns.countplot(x="Sector", data=companies)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.ylabel("No of Count")
plt.xlabel("Sector")
plt.title("Sector by Count")
plt.tight_layout()
plt.show()

#General working on S&P500 Companies dataset
names=['Apple INC.', 'eBay Inc', 'Gartner Inc']
print(names)
prices=[155.15, 41.02, 114.26]
print(prices)
print(names[0])
print(names[1])
print(prices[-1])
names_subset = names[1:3]
print(names_subset)
prices_subset = prices[0:2]
print(prices_subset)
# Create and print the nested list stocks
stocks = [['Apple INC.', 'eBay Inc', 'Gartner Inc'],[155.15, 41.02, 114.26]]
print(stocks)
# Use list indexing to obtain the list of prices
print(stocks[1])
# Use indexing to obtain company name eBay Inc
print(stocks[0][1])
# Use indexing to obtain 114.26
print(stocks[1][2])
# Print the sorted list prices
prices = [155.15, 41.02, 114.26]
prices.sort()
print(prices)
# Find the maximum price in the list price
prices = [155.15, 41.02, 114.26]
price_max = max(prices)
print(price_max)
# Append a name to the list names
names.append('Campbell Soup')
print(names)
# Extend list names
more_elements = ['Exelon Corp.', 'FMC Corporation']
names.extend(more_elements)
print(names)
max_price = max(prices)
# Identify index of max price
max_index = prices.index(max_price)
# Identify the name of the company with max price
max_stock_name = names[max_index]
# Fill in the blanks
print('The largest stock price is associated with ' + max_stock_name + ' and is $' + str(max_price) + '.')

#NUMPY workings
# Lists
prices = [155.15,41.02,114.26,44.83,35.98,80.87]
earnings = [9.2,-1.07,2.31,2.89,1.23,1.56]
# NumPy arrays
prices_array = np.array(prices)
earnings_array = np.array(earnings)
# Print the arrays
print(prices_array)
print(earnings_array)
# Create PE ratio array
pe_array = prices_array/earnings_array
# Print pe_array
print(pe_array)
# Subset the first three elements
prices_subset_1 = prices_array[0:3]
print(prices_subset_1)
# Subset every third element
prices_subset_3 = prices_array[0:7:3]
print(prices_subset_3)
# Create a 2D array of prices and earnings
stock_array = np.array([prices , earnings])
print(stock_array)
# Print the shape of stock_array
print(stock_array.shape)
# Print the size of stock_array
print(stock_array.size)
# Transpose stock_array
stock_array_transposed = np.transpose(stock_array)
print(stock_array_transposed)
# Print the shape of stock_array
print(stock_array_transposed.shape)
# Print the size of stock_array
print(stock_array_transposed.size)
# Subset prices from stock_array_transposed
prices = stock_array_transposed[:,0]
print(prices)
# Subset earnings from stock_array_transposed
earnings = stock_array_transposed[:, 1]
print(earnings)
# Subset the price and earning for first company
company_1 = stock_array_transposed[0, :]
print(company_1)
# Calculate the mean
prices_mean = np.mean(prices)
print(prices_mean)
# Calculate the standard deviation
prices_std = np.std(prices)
print(prices_std)
# Create and print company IDs
company_ids = np.arange(1, 8, 1)
print(company_ids)
# Use array slicing to select specific company IDs
company_ids_odd = np.arange (1, 9, 2)
print(company_ids_odd)
# Find the mean
price_mean = np.mean(prices)

# Create boolean array
boolean_array = (prices > price_mean)
print(boolean_array)
# Select prices that are greater than average
above_avg = prices[boolean_array]
print(above_avg)
#API workings
import requests
data=requests.get("http://api.open-notify.org/iss-now.json")
data2=data.json()
print(data2["timestamp"])
print(data2)
#No of people in space & names
data=requests.get("http://api.open-notify.org/astros.json")
data3=data.json()
print(data3["number"])
for p in data3["people"]:
    print(p["name"])
#stock market API
stockdata=requests.get("https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=demo")
datastk=stockdata.json()
print(datastk)
print(datastk["Meta Data"])
