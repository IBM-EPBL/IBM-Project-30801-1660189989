#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_excel('Global_Superstore2.xlsx')


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.shape


# In[ ]:


#We can see that the data has 51290 data entries with 24 features.


# In[7]:


# can see that there is negetive trend in profit
# in sales  data is either right skewed or there are too many outliers
# the discount falls in the range of 75% - 100%
df.describe()


# In[ ]:


# We can see that only the postal code attribute has 41,296 null vlues. Thats almost 80 % of the values in the column are null


# In[8]:


df.isnull().sum()


# In[9]:


df.isnull().sum()


# In[ ]:


#lets understand how many of them are categorical and how many unique values each categorical columns have,


# In[10]:


cat_cols = df.select_dtypes(exclude=['int64','float64']).columns
cat_cols


# In[11]:


# num_cols = df.select_dtypes(include=['int64', 'float64']).columns
num_colms = df._get_numeric_data()
num_colms.head()
# df['Ship Mode'].unique()
# df['Ship Mode'].value_counts()
# df['Segment'].value_counts()
# df['Market'].value_counts()
# df['Region'].value_counts()
# df['Category'].value_counts()
# df['Sub-Category'].value_counts()
# df['Product Name'].value_counts()
# df['Order Priority'].value_counts()


# In[12]:


#Customer Analysis
#1. Profile the customers based on their frequency of purchase - calculate frequency of purchase for each customer and plot a histogram to get the thresholdfor Low/Mid/High frequency customers


# In[13]:


# purchase frequency = no of orders / no of unique customers (365 days)
# df['Order ID'].groupby('Customer ID').count()
df.nunique()


# In[14]:


df_customer = df[['Customer ID','Order ID','Order Date', 'Ship Date', 'Ship Mode','Country']]
# .drop_duplicates()
df_customer.count()


# In[15]:


df_customer1 = df[['Customer ID','Order ID','Order Date', 'Ship Date', 'Ship Mode','Country']].drop_duplicates()
df_customer1.count()


# In[16]:


# df2 = df.drop_duplicates()
# df2.shape
# df_customer[[df_customer['Customer ID','Order ID','Order Date']].duplicated() == True]
# df_customer['Customer ID'].nunique()
df_customer1.head()


# In[17]:


df_customer1['Customer ID'].nunique()
# there are only 1590 unique customer ids out of 25754 rows in the customer data


# In[18]:


# First_Purchase_Date = pd.to_datetime(df['Order Date']).min()
# latest_purchase_date = pd.to_datetime(df['Order Date']).max()
# Total_shipping_cost = df['Shipping Cost'].sum()
# total_sales = df['Sales'].sum()
# unique_order_id = df['Order_id'].nunique()


def new_features(x):
    d = []
    d.append(x['Order ID'].nunique())
    d.append(x['Sales'].sum())
    d.append(x['Shipping Cost'].sum())
    d.append(pd.to_datetime(x['Order Date']).min())
    d.append(pd.to_datetime(x['Order Date']).max())
    d.append(x['City'].nunique())
    return pd.Series(d, index=['Purchases','Total_Sales','Total_Cost','First_Purchase_Date','Latest_Purchase_Date','Location_Count'])

df_customer_new = df.groupby('Customer ID').apply(new_features)


# In[19]:


df_customer_new.columns


# In[20]:


from datetime import datetime
df_customer_new['Duration'] = (df_customer_new['Latest_Purchase_Date'] - df_customer_new['First_Purchase_Date']).dt.days


# In[21]:


df_customer_new['Frequency'] = df_customer_new['Duration']/df_customer_new['Purchases']
df_customer_new['Frequency'].head()


# In[22]:


df_customer_new.head()


# In[23]:


df_customer_new['Frequency'].describe()


# In[24]:


# df_customer_new['Duration'].describe()
# Total time duration = approx 3.7 years


# In[25]:


# 1. Profile the customers based on their frequency of purchase - calculate frequency of purchase for
# each customer and plot a histogram to get the threshold for Low/Mid/High frequency customers

plt.hist(df_customer_new['Frequency'], bins=3)
plt.xlabel('frequency of purchase')
plt.show()


# In[26]:


# bucketing continuous data
def freq(x):
    if x < 219:
        return 'Low' 
    elif x < 436:
        return 'Mid' 
    else:
        return 'High'

df_customer_new['freq_range'] = df_customer_new.Frequency.apply(freq)

df_customer_new['freq_range'].value_counts()


# In[27]:


# profiling based on purchase frequency
df_customer_new.head()


# In[28]:


# Are the high frequent customers contributing more revenue


# In[29]:


result = df.groupby(["Customer Name"])['Profit'].aggregate(np.sum).reset_index().sort_values('Profit',ascending = False).head(20)


# In[30]:


# revenue = sales *  quantity 


# In[31]:


df_customer_new['Revenue'] = df_customer_new['Purchases'] *  df_customer_new['Total_Sales']


# In[32]:


sns.catplot(x="freq_range", y="Revenue", kind="bar", data=df_customer_new);


# In[33]:


#Are they also profitable - what is the profit margin across the buckets


# In[34]:


profit_margin = df_customer_new.groupby('freq_range')


# In[35]:


profit_margin


# In[36]:


# Which customer segment is most profitable in each year ( there is a column called customer segment


# In[37]:


df.Segment.value_counts()


# In[38]:


sns.countplot(x="Segment",data = df)


# In[39]:


df_x = df
df_x['year'] = pd.DatetimeIndex(df['Order Date']).year
df_x.head()


# In[40]:


plt.figure(figsize=(24,15))
sns.catplot(x="Segment", col="year", data=df_x, kind="count")
plt.show()


# In[41]:


# How the customers are distributed across the countries - pie chart 


# In[42]:


customer_country = pd.DataFrame({'Count' : df.groupby(["Country"]).size()}).reset_index().sort_values('Count',ascending = False).head(10)


# In[43]:


customer_country


# In[44]:


from matplotlib import pyplot as plt
import numpy as np
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
Countries = customer_country['Country']
cust_count = customer_country['Count']
ax.pie(cust_count, labels = Countries,autopct='%1.2f%%')
plt.show()


# In[45]:


#Write a function to split the global store data into different unique data frames based on the unique values in country column [Means, we should have one data frame for one country as function output ]


# In[46]:


# def new_dataframe(x):
#     d = []
#     return pd.Series(d, index= df.columns)


# df_new_dataframe = df.groupby('Country').apply(new_dataframe)

for i, g in df.groupby('Country'):
    globals()['df_' + str(i)] =  g
    
print (df_China)


# In[47]:


grouped = df.groupby(df.Country)

China = grouped.get_group("China")


# In[48]:


#Product Analysis
#Plot the countries with their total sales - bar plot


# In[49]:


plt.figure(figsize=(16,8))
countries = df.groupby('Country')['Sales'].count().sort_values(ascending=False)
countries = countries [:60]
countries.plot(kind='bar', color='orange')
plt.title('Top 60 Countries in Sales')
plt.ylabel('Sales')
plt.xlabel('Countries')


# In[50]:


#What are top 5 profit making product types on a yearly basis


# In[51]:


products = df.groupby('Product Name')['Profit'].count().sort_values(ascending=False)
# top_5_products = products[:5]
# top_5_products
products.head()


# In[52]:


# What is the average delivery date across the countries - bar plot


# In[53]:


df_x = df.drop_duplicates()

df_x['Order_to_Ship_Days'] = (pd.to_datetime(df_x['Ship Date']) 
                                           - pd.to_datetime(df_x['Order Date'])).dt.days

# # df_customer1.head()
#days_taken = df_x.groupby('Order ID')['Order_to_Ship_Days'].mean()
# total_days = days_taken.to_frame()
# total_days.head()


# In[54]:


df_x.head()


# In[55]:


plt.figure(figsize=(16,8))
countries = df_x.groupby('Country')['Order_to_Ship_Days'].mean().sort_values(ascending=False)
countries = countries [:60]
countries.plot(kind='bar', color='red')
plt.title('Top 60 Countries in Sales')
plt.ylabel('Average shipment date')
plt.xlabel('Countries')


# In[ ]:




