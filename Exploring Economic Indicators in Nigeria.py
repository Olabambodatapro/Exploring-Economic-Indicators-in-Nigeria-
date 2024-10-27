#!/usr/bin/env python
# coding: utf-8

# In[58]:


# import pandas, matplotlib and seaborn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset
file_path = "Nigeria_economy(small).csv"
df = pd.read_csv(file_path)

# Display first few rows to get an overview
print(df.head())

# Get information about data types and missing values
df.info()

# Statistical summary of numerical columns
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Check column names
print(df.columns)

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# sketching graph for GDP and GDP growth from 1990 to 2023

# Assuming df is already loaded
GDP_data = df[df['Series Name'] == 'GDP (current US$)']

# Extract years (x-axis) and GDP values (y-axis)
years = ['1990 [YR1990]', '2000 [YR2000]', '2014 [YR2014]', '2015 [YR2015]', 
         '2016 [YR2016]', '2017 [YR2017]', '2018 [YR2018]', '2019 [YR2019]', 
         '2020 [YR2020]', '2021 [YR2021]', '2022 [YR2022]', '2023 [YR2023]']
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#Graph for GDP
# Ensure GDP values are converted to floats
GDP_Yearly = GDP_data[years].values.flatten().astype(float)

# Extract only the year portion as integers
years_numeric = [int(year.split()[0]) for year in years]

# Plot the data
plt.plot(years_numeric, GDP_Yearly, marker='o')
plt.title('GDP Over the Years')
plt.xlabel('Year')
plt.ylabel('GDP (current US$)')

# Set custom y-axis limits based on actual data range
plt.ylim(5e+10, 6e+11)  # Adjust according to the data range

# Use MaxNLocator to control number of y-axis ticks
ax = plt.gca()
ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=10))

# Show the plot
plt.show()


# In[43]:


#Graph for GDP Growth
# Assuming df is already loaded
GDP_Growth_data = df[df['Series Name'] == 'GDP growth (annual %)']
# Ensure GDP values are converted to floats
GDP_Growth_Yearly = GDP_Growth_data[years].values.flatten().astype(float)

# Extract only the year portion as integers
years_numeric = [int(year.split()[0]) for year in years]

# Plot the data
plt.plot(years_numeric, GDP_Growth_Yearly, marker='o')
plt.title('GDP growth (annual%) Over the Years')
plt.xlabel('Year')
plt.ylabel('GDP growth (annual%)')

# Set custom y-axis limits based on actual data range
plt.ylim(-2, 12)  # Adjust according to the data range

# Use MaxNLocator to control number of y-axis ticks
ax = plt.gca()
ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=10))

# Show the plot
plt.show()


# In[42]:


#Graph for GDP Growth
# Assuming df is already loaded
inflation_percentage_data = df[df['Series Name'] == 'Inflation, GDP deflator (annual %)']
# Ensure GDP values are converted to floats
Inflation_percentage_Yearly = inflation_percentage_data[years].values.flatten().astype(float)

# Extract only the year portion as integers
years_numeric = [int(year.split()[0]) for year in years]

# Plot the data
plt.plot(years_numeric, Inflation_percentage_Yearly, marker='o')
plt.title('inflation (annual%) Over the Years')
plt.xlabel('Year')
plt.ylabel('inflation (annual%)')

# Set custom y-axis limits based on actual data range
plt.ylim(0, 24)  # Adjust according to the data range

# Use MaxNLocator to control number of y-axis ticks
ax = plt.gca()
ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=10))

# Show the plot
plt.show()


# In[46]:


#Graph for population
# Assuming df is already loaded
population_data = df[df['Series Name'] == 'Population, total']
# Ensure GDP values are converted to floats
Pop_total_Yearly = population_data[years].values.flatten().astype(float)

# Extract only the year portion as integers
years_numeric = [int(year.split()[0]) for year in years]

# Plot the data
plt.plot(years_numeric, Pop_total_Yearly, marker='o')
plt.title('Population Over the Years')
plt.xlabel('Year')
plt.ylabel('Population')

# Set custom y-axis limits based on actual data range
plt.ylim(90000000, 230000000)  # Adjust according to the data range

# Use MaxNLocator to control number of y-axis ticks
ax = plt.gca()
ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=10))

# Show the plot
plt.show()


# In[70]:


#Train Test Split
# Define the features (Inflation, Population) and the target variable (GDP)
X = Pop_total_Yearly, Inflation_percentage_Yearly
y = GDP_Yearly

X = np.array([X[0], X[1]]).T  # Transpose to make it 2D with shape (n_samples, n_features)

## Ensure to select the correct columns for X and y
print("X:", X)
print("y:", y)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[71]:


# Reshape X_train and y_train to make sure they're 2D arrays
X_train = X_train.reshape(-1, 1) if X_train.ndim == 1 else X_train
y_train = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)


# In[72]:


#Evaluate model preformance
# Reshape X_test to ensure it is a 2D array
X_test = X_test.reshape(-1, 1) if X_test.ndim == 1 else X_test

# Predict GDP for the test set
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: {rmse:.2f}")


# In[74]:


# Hypothetical values for inflation and population for 2025
hypothetical_inflation_rate_2025 = 15.0  # Example projected inflation rate
hypothetical_population_2025 = 350e6       # Example projected population (e.g., 350 million)

# Create future_data with both features for 2025
future_data_2025 = np.array([[hypothetical_inflation_rate_2025, hypothetical_population_2025]])

# Predict GDP for 2025
future_gdp_pred_2025 = model.predict(future_data_2025)

# Print the prediction as a float using the first element of the prediction array
print(f"Predicted GDP for 2025: ${float(future_gdp_pred_2025[0]):.6e}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




