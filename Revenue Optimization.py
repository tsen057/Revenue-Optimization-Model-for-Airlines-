#!/usr/bin/env python
# coding: utf-8

# In[114]:


import pandas as pd
import numpy as np
from datetime import datetime

# Load datasets
airport_info_df = pd.read_csv("C:/Users/tejas/OneDrive/Desktop/flight dataset/aiport_info.csv")  # Airport info data
airline_key_df = pd.read_csv("C:/Users/tejas/OneDrive/Desktop/flight dataset/airline_key.csv")  # Airline key data
alljoined_airlines_df = pd.read_csv(
    "C:/Users/tejas/OneDrive/Desktop/flight dataset/alljoined_airlines.csv", 
    dtype={'CANCELLATION_CODE': str})

# Load ticket pricing data 
ticket_pricing_data = []
for year in range(2018, 2022): 
    file_path = f'C:/Users/tejas/OneDrive/Desktop/flight dataset/AverageFare_Annual_{year}.csv'
    df = pd.read_csv(file_path)
    df['Year'] = year 
    ticket_pricing_data.append(df)

# Combine all years into one DataFrame
ticket_pricing_df = pd.concat(ticket_pricing_data, ignore_index=True)

# Convert date format in flight data
alljoined_airlines_df['FL_DATE'] = pd.to_datetime(alljoined_airlines_df['FL_DATE'], errors='coerce')
alljoined_airlines_df.dropna(subset=['FL_DATE'], inplace=True)  # Remove invalid dates

# Filter the data for the years 2018-2022
alljoined_airlines_df = alljoined_airlines_df[alljoined_airlines_df['FL_DATE'].dt.year.between(2018, 2022)]

# Merge airport codes
alljoined_airlines_df = pd.merge(alljoined_airlines_df, airport_info_df[['Code.y', 'ORGIN_AIPORT_ID']], 
                                 left_on='ORIGIN_AIRPORT_ID', right_on='ORGIN_AIPORT_ID', how='left')

alljoined_airlines_df = pd.merge(alljoined_airlines_df, airport_info_df[['Code.y', 'ORGIN_AIPORT_ID']],
                                 left_on='DEST_AIRPORT_ID', right_on='ORGIN_AIPORT_ID', how='left')

# Replace airport IDs with codes and handle missing values
alljoined_airlines_df['ORIGIN_AIRPORT_ID'] = alljoined_airlines_df['Code.y_x'].fillna('Unknown')
alljoined_airlines_df['DEST_AIRPORT_ID'] = alljoined_airlines_df['Code.y_y'].fillna('Unknown')

# Drop temporary columns
alljoined_airlines_df.drop(columns=['Code.y_x', 'Code.y_y'], inplace=True)

# Create a column for month-year
alljoined_airlines_df['Month_Year'] = alljoined_airlines_df['FL_DATE'].dt.to_period('M')

# Calculate statistics
route_flight_counts = alljoined_airlines_df.groupby(['Month_Year', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID']).size().reset_index(name='flight_count')
avg_delays = alljoined_airlines_df.groupby(['Month_Year', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID'])[['DEP_DELAY', 'ARR_DELAY']].mean().reset_index()
cancellation_rate = alljoined_airlines_df.groupby(['Month_Year', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID'])['CANCELLED'].mean().reset_index()

# Merge statistics
route_data = pd.merge(route_flight_counts, avg_delays, on=['Month_Year', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID'], how='left')
route_data = pd.merge(route_data, cancellation_rate, on=['Month_Year', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID'], how='left')

# Fill missing values in delays and cancellations
route_data[['DEP_DELAY', 'ARR_DELAY', 'CANCELLED']] = route_data[['DEP_DELAY', 'ARR_DELAY', 'CANCELLED']].fillna(0)

# Estimate demand
route_data['estimated_demand'] = route_data['flight_count'] * (1 - route_data['CANCELLED'])

# Prepare ticket pricing data
ticket_pricing_df['Year'] = ticket_pricing_df['Year'].astype(str).str[:4].astype(int)
ticket_pricing_df['Month_Year'] = pd.to_datetime(ticket_pricing_df['Year'].astype(str) + '-01').dt.to_period('M')
ticket_pricing_df['Airport Code'] = ticket_pricing_df['Airport Code'].astype(str)
route_data['ORIGIN_AIRPORT_ID'] = route_data['ORIGIN_AIRPORT_ID'].astype(str)
ticket_pricing_df.rename(columns={'Airport Code': 'ORIGIN_AIRPORT_ID'}, inplace=True)

# Merge ticket pricing data for origin airport
route_data = pd.merge(route_data, ticket_pricing_df[['Month_Year', 'ORIGIN_AIRPORT_ID', 'Average Fare ($)']],
                      on=['Month_Year', 'ORIGIN_AIRPORT_ID'], how='left', suffixes=('', '_origin'))

# Merge ticket pricing data for destination airport
route_data = pd.merge(route_data, ticket_pricing_df[['Month_Year', 'ORIGIN_AIRPORT_ID', 'Average Fare ($)']],
                      left_on=['Month_Year', 'DEST_AIRPORT_ID'], right_on=['Month_Year', 'ORIGIN_AIRPORT_ID'], 
                      how='left', suffixes=('_origin', '_dest'))


# Filter rows where both origin and destination have available fare data since some routes' ticket data is not present
route_data = route_data[route_data['Average Fare ($)_origin'].notnull() | route_data['Average Fare ($)_dest'].notnull()]

# Exclude rows where the destination airport is 'HIK' or 'BSM' since there is not ticket data for these specific airports
route_data = route_data[~route_data['DEST_AIRPORT_ID'].isin(['HIK', 'BSM'])]
route_data = route_data[~route_data['ORIGIN_AIRPORT_ID_origin'].isin(['HIK', 'BSM'])]


# Adjust demand based on pricing
route_data['adjusted_demand'] = route_data['estimated_demand'] * (1 / (1 + route_data['Average Fare ($)_origin'] / route_data['Average Fare ($)_origin'].max())) *                                  (1 / (1 + route_data['Average Fare ($)_dest'] / route_data['Average Fare ($)_dest'].max()))

# Check for remaining missing values
print(route_data.isnull().sum())

# Export the data
route_data.to_csv('C:/Users/tejas/OneDrive/Desktop/flight dataset/adjusted_demand_by_route.csv', index=False)

# Merging route data with ticket pricing data based on Month_Year and ORIGIN_AIRPORT_ID
common_routes = pd.merge(route_data, ticket_pricing_df,
                         left_on=['Month_Year', 'ORIGIN_AIRPORT_ID_origin'],
                         right_on=['Month_Year', 'ORIGIN_AIRPORT_ID'],
                         how='inner')

# Verify the merged data
print(f"Merged DataFrame head:\n{common_routes.head()}")

# Check the matching route count
print(f"Matching routes: {len(common_routes)} / {len(route_data)}")
print(f"Percentage of routes with fare data: {len(common_routes) / len(route_data) * 100:.2f}%")


# In[118]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans
from datetime import datetime

# Load datasets
airport_info_df = pd.read_csv("C:/Users/tejas/OneDrive/Desktop/flight dataset/aiport_info.csv")
airline_key_df = pd.read_csv("C:/Users/tejas/OneDrive/Desktop/flight dataset/airline_key.csv")
alljoined_airlines_df = pd.read_csv("C:/Users/tejas/OneDrive/Desktop/flight dataset/alljoined_airlines.csv", dtype={'CANCELLATION_CODE': str})

ticket_pricing_data = []
for year in range(2018, 2022): 
    file_path = f'C:/Users/tejas/OneDrive/Desktop/flight dataset/AverageFare_Annual_{year}.csv'
    df = pd.read_csv(file_path)
    df['Year'] = year 
    ticket_pricing_data.append(df)

ticket_pricing_df = pd.concat(ticket_pricing_data, ignore_index=True)

# Convert date format in flight data
alljoined_airlines_df['FL_DATE'] = pd.to_datetime(alljoined_airlines_df['FL_DATE'], errors='coerce')
alljoined_airlines_df.dropna(subset=['FL_DATE'], inplace=True)
alljoined_airlines_df = alljoined_airlines_df[alljoined_airlines_df['FL_DATE'].dt.year.between(2018, 2022)]

# Merge airport codes
alljoined_airlines_df = pd.merge(alljoined_airlines_df, airport_info_df[['Code.y', 'ORGIN_AIPORT_ID']], 
                                 left_on='ORIGIN_AIRPORT_ID', right_on='ORGIN_AIPORT_ID', how='left')

alljoined_airlines_df = pd.merge(alljoined_airlines_df, airport_info_df[['Code.y', 'ORGIN_AIPORT_ID']],
                                 left_on='DEST_AIRPORT_ID', right_on='ORGIN_AIPORT_ID', how='left')

# Replace airport IDs with codes
alljoined_airlines_df['ORIGIN_AIRPORT_ID'] = alljoined_airlines_df['Code.y_x'].fillna('Unknown')
alljoined_airlines_df['DEST_AIRPORT_ID'] = alljoined_airlines_df['Code.y_y'].fillna('Unknown')
alljoined_airlines_df.drop(columns=['Code.y_x', 'Code.y_y'], inplace=True)

# Create month-year column
alljoined_airlines_df['Month_Year'] = alljoined_airlines_df['FL_DATE'].dt.to_period('M')

# Aggregate statistics
route_flight_counts = alljoined_airlines_df.groupby(['Month_Year', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID']).size().reset_index(name='flight_count')

avg_delays = alljoined_airlines_df.groupby(['Month_Year', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID'])[['DEP_DELAY', 'ARR_DELAY']].mean().reset_index()

cancellation_rate = alljoined_airlines_df.groupby(['Month_Year', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID'])['CANCELLED'].mean().reset_index()

# Merge statistics
route_data = pd.merge(route_flight_counts, avg_delays, on=['Month_Year', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID'], how='left')
route_data = pd.merge(route_data, cancellation_rate, on=['Month_Year', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID'], how='left')
route_data.fillna(0, inplace=True)
route_data['estimated_demand'] = route_data['flight_count'] * (1 - route_data['CANCELLED'])

# Price optimization using elasticity modeling
def price_elasticity(demand, price):
    elasticity = -1.5  # Assumed elasticity coefficient
    return demand * (price ** elasticity)

route_data['optimized_demand'] = price_elasticity(route_data['estimated_demand'], route_data['flight_count'])

# Demand Forecasting (ARIMA Model)
def forecast_demand(data):
    model = ARIMA(data, order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=12)
    return forecast

forecast_results = forecast_demand(route_data['estimated_demand'])

# Dynamic Route Management using K-Means
num_clusters = 5
km = KMeans(n_clusters=num_clusters, random_state=42)
route_data['cluster'] = km.fit_predict(route_data[['estimated_demand', 'flight_count']])

# Export to Power BI
route_data.to_csv('C:/Users/tejas/OneDrive/Desktop/flight dataset/optimized_routes.csv', index=False)

# Visualization
# Ensure your dataset is loaded correctly
print(route_data.head())  # Check if 'date' column exists

# Convert date column to datetime format
route_data['date'] = pd.to_datetime(route_data['Month_Year'], errors='coerce')

# Extract year from the date column
route_data['year'] = route_data['date'].dt.year

# Aggregate demand data by year
yearly_demand = route_data.groupby('year')['estimated_demand'].sum().reset_index()

# Plot demand forecasting with yearly aggregation
plt.figure(figsize=(10, 5))
sns.lineplot(x='year', y='estimated_demand', data=yearly_demand, marker='o')
plt.title('Yearly Demand Forecasting')
plt.xlabel('Year')
plt.ylabel('Estimated Demand')
plt.grid(True)
plt.show()


plt.figure(figsize=(10,5))
sns.scatterplot(x=route_data['flight_count'], y=route_data['optimized_demand'], hue=route_data['cluster'])
plt.xlabel('Number of Flights')
plt.ylabel('Optimized Demand')
plt.title('Price Elasticity Curve')
plt.show()

route_pivot = route_data.groupby(['ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID'])['estimated_demand'].sum().unstack(fill_value=0)
sns.heatmap(route_pivot, cmap='coolwarm')
plt.title('Route Performance Heatmap')
plt.show()


# In[ ]:




