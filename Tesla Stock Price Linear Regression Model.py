#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().system('pip install chart_studio')
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot


# In[3]:


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


# In[4]:


tesla_df = pd.read_csv('Tesla Stock Price (2010 to 2023).csv')


# In[5]:


tesla_df.info()


# In[6]:


tesla_df.head()


# In[15]:


tesla_df['Date'] = pd.to_datetime(tesla_df['Date'], format='%d/%m/%Y')


# In[16]:


tesla_df['Date'] = pd.to_datetime(tesla_df['Date'])


# In[18]:


print(f'Dataframe contains stock prices between {tesla_df.Date.min()} {tesla_df.Date.max()}')
print(f'Total days = {(tesla_df.Date.max() - tesla_df.Date.min()).days} days')


# In[19]:


tesla_df.describe()


# In[20]:


tesla_df[['Open', 'High', 'Low', 'Close', 'Adj Close']].plot(kind='box')


# In[24]:


import plotly.graph_objs as go
from plotly.offline import iplot

# Assuming you have already loaded your data into the tesla_df DataFrame

# Create a Scatter trace using the Date and Close columns from the DataFrame
scatter_trace = go.Scatter(x=tesla_df['Date'], y=tesla_df['Close'], mode='lines', name='Tesla Stock Price')

# Define the layout
layout = go.Layout(
    title='Stock Price of Tesla',
    xaxis=dict(
        title='Date',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Price',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)

# Create the Plotly figure with the Scatter trace and layout
tesla_data = [{'x':tesla_df['Date'], 'y':tesla_df['Close']}]
plot = go.Figure(data=[scatter_trace], layout=layout)




# In[25]:


# Display the plot
iplot(plot)


# In[27]:


#Building the regression model


# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


#For preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
# For model evaluation
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score


# In[31]:


# Split the dataset into train and test sets
X = np.array(tesla_df.index).reshape(-1,1)
Y = tesla_df['Close']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)


# In[32]:


# Featuring Scaling
scaler = StandardScaler().fit(X_train)


# In[33]:


from sklearn.linear_model import LinearRegression


# In[34]:


# Creating a linear model
lm = LinearRegression()
lm.fit(X_train, Y_train)


# In[40]:


# Plot actual and predicted values for train dataset
trace0 = go.Scatter(
    x = X_train.T[0],
    y = Y_train,
    mode = 'markers',
    name = 'Actual'
)
trace1 = go.Scatter(
    x = X_train.T[0],
    y = lm.predict(X_train).T,
    mode = 'lines',
    name = 'Predicted'
)
tesla_data = [trace0, trace1]
layout.xaxis.title.text = 'Day'
plot2 = go.Figure(data = tesla_data, layout=layout)


# In[41]:


iplot(plot2)


# In[44]:


# Calculate scores for model evaluation
scores = f'''
{'Metric'.ljust(10)}{'Train'.center(20)}{'Test'.center(20)}
{'r2_score'.ljust(10)}{r2_score(Y_train, lm.predict(X_train))}\t{r2_score(Y_test, lm.predict(X_test))}
{'MSE'.ljust(10)}{mse(Y_train, lm.predict(X_train))}\t{r2_score(Y_test, lm.predict(X_test))}
'''
print(scores)


# In[ ]:




