import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
airport_info_df = pd.read_csv("C:/Users/tejas/OneDrive/Desktop/flight dataset/aiport_info.csv")
airline_key_df = pd.read_csv("C:/Users/tejas/OneDrive/Desktop/flight dataset/airline_key.csv")
alljoined_airlines_df = pd.read_csv("C:/Users/tejas/OneDrive/Desktop/flight dataset/alljoined_airlines.csv", dtype={'CANCELLATION_CODE': str})

# Load ticket pricing data
ticket_pricing_data = []
for year in range(2018, 2022):
    file_path = f'C:/Users/tejas/OneDrive/Desktop/flight dataset/AverageFare_Annual_{year}.csv'
    df = pd.read_csv(file_path)
    df['Year'] = year
    ticket_pricing_data.append(df)

ticket_pricing_df = pd.concat(ticket_pricing_data, ignore_index=True)
ticket_pricing_df['Year'] = ticket_pricing_df['Year'].astype(str).str[:4].astype(int)
ticket_pricing_df['Month_Year'] = pd.to_datetime(ticket_pricing_df['Year'].astype(str) + '-01').dt.to_period('M')
ticket_pricing_df.rename(columns={'Airport Code': 'ORIGIN_AIRPORT_ID'}, inplace=True)
ticket_pricing_df['ORIGIN_AIRPORT_ID'] = ticket_pricing_df['ORIGIN_AIRPORT_ID'].astype(str)

# Preprocess flight data
alljoined_airlines_df['FL_DATE'] = pd.to_datetime(alljoined_airlines_df['FL_DATE'], errors='coerce')
alljoined_airlines_df.dropna(subset=['FL_DATE'], inplace=True)
alljoined_airlines_df = alljoined_airlines_df[alljoined_airlines_df['FL_DATE'].dt.year.between(2018, 2022)]

# Merge airport codes
alljoined_airlines_df = pd.merge(alljoined_airlines_df, airport_info_df[['Code.y', 'ORGIN_AIPORT_ID']], left_on='ORIGIN_AIRPORT_ID', right_on='ORGIN_AIPORT_ID', how='left')
alljoined_airlines_df = pd.merge(alljoined_airlines_df, airport_info_df[['Code.y', 'ORGIN_AIPORT_ID']], left_on='DEST_AIRPORT_ID', right_on='ORGIN_AIPORT_ID', how='left')
alljoined_airlines_df['ORIGIN_AIRPORT_ID'] = alljoined_airlines_df['Code.y_x'].fillna('Unknown')
alljoined_airlines_df['DEST_AIRPORT_ID'] = alljoined_airlines_df['Code.y_y'].fillna('Unknown')
alljoined_airlines_df.drop(columns=['Code.y_x', 'Code.y_y'], inplace=True)

# Monthly data
alljoined_airlines_df['Month_Year'] = alljoined_airlines_df['FL_DATE'].dt.to_period('M')

# Route statistics
route_flight_counts = alljoined_airlines_df.groupby(['Month_Year', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID']).size().reset_index(name='flight_count')
avg_delays = alljoined_airlines_df.groupby(['Month_Year', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID'])[['DEP_DELAY', 'ARR_DELAY']].mean().reset_index()
cancellation_rate = alljoined_airlines_df.groupby(['Month_Year', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID'])['CANCELLED'].mean().reset_index()

route_data = pd.merge(route_flight_counts, avg_delays, on=['Month_Year', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID'], how='left')
route_data = pd.merge(route_data, cancellation_rate, on=['Month_Year', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID'], how='left')
route_data.fillna(0, inplace=True)
route_data['estimated_demand'] = route_data['flight_count'] * (1 - route_data['CANCELLED'])

# Merge fare data
route_data['ORIGIN_AIRPORT_ID'] = route_data['ORIGIN_AIRPORT_ID'].astype(str)
route_data['DEST_AIRPORT_ID'] = route_data['DEST_AIRPORT_ID'].astype(str)

route_data = pd.merge(route_data, ticket_pricing_df[['Month_Year', 'ORIGIN_AIRPORT_ID', 'Average Fare ($)']],
                      on=['Month_Year', 'ORIGIN_AIRPORT_ID'], how='left', suffixes=('', '_origin'))

route_data = pd.merge(route_data, ticket_pricing_df[['Month_Year', 'ORIGIN_AIRPORT_ID', 'Average Fare ($)']],
                      left_on=['Month_Year', 'DEST_AIRPORT_ID'],
                      right_on=['Month_Year', 'ORIGIN_AIRPORT_ID'], how='left', suffixes=('_origin', '_dest'))

# Remove airports without pricing info
route_data = route_data[~route_data['DEST_AIRPORT_ID'].isin(['HIK', 'BSM'])]
route_data = route_data[route_data['Average Fare ($)_origin'].notnull() | route_data['Average Fare ($)_dest'].notnull()]

# --- Log-Linear Price Elasticity Model ---
def price_elasticity(demand, price, elasticity=-0.1):
    return demand * np.exp(elasticity * price)

# Create price proxy and handle missing
route_data['price_proxy'] = (route_data[['Average Fare ($)_origin', 'Average Fare ($)_dest']].mean(axis=1))
non_zero_mean = route_data['price_proxy'].replace(0, pd.NA).dropna().mean()
route_data['price_proxy'] = route_data['price_proxy'].fillna(non_zero_mean)
route_data['price_proxy'] = route_data['price_proxy'].replace(0, non_zero_mean)

# Apply elasticity model
#route_data['optimized_demand'] = price_elasticity(route_data['estimated_demand'], route_data['price_proxy'] / 100)
route_data['optimized_demand'] = price_elasticity(route_data['estimated_demand'], route_data['price_proxy'] / 100, elasticity=0.02)


# Add date and year columns
route_data['date'] = route_data['Month_Year'].dt.to_timestamp()
route_data['year'] = route_data['date'].dt.year

# --- Forecasting Based on Annual Demand ---
annual_demand_series = route_data.groupby('year')['estimated_demand'].sum()
annual_model = ARIMA(annual_demand_series, order=(1, 1, 0))
annual_model_fit = annual_model.fit()
annual_forecast = annual_model_fit.forecast(steps=2)
annual_forecast.index = [annual_demand_series.index.max() + i for i in range(1, 3)]
forecast_annual_df = annual_forecast.to_frame(name='forecasted_annual_demand')
print("Forecasted Annual Demand:\n", forecast_annual_df)

# --- Revenue uplift calculation ---
route_data['original_revenue'] = route_data['estimated_demand'] * route_data['price_proxy']
route_data['optimized_revenue'] = route_data['optimized_demand'] * route_data['price_proxy']
route_data['revenue_uplift_percent'] = ((route_data['optimized_revenue'] - route_data['original_revenue']) / route_data['original_revenue']) * 100
avg_uplift = route_data['revenue_uplift_percent'].mean()
print(f"Average projected revenue uplift: {avg_uplift:.2f}%")

# Cluster routes for dynamic management
km = KMeans(n_clusters=5, random_state=42)
route_data['cluster'] = km.fit_predict(route_data[['estimated_demand', 'flight_count']])

# --- Prepare export for Power BI ---
# Explicitly cast data types before export
summary_for_powerbi = route_data[[
    'Month_Year', 'ORIGIN_AIRPORT_ID_origin', 'DEST_AIRPORT_ID',
    'estimated_demand', 'optimized_demand', 'price_proxy',
    'original_revenue', 'optimized_revenue', 'revenue_uplift_percent',
    'flight_count', 'cluster'
]].copy()

# Set types
summary_for_powerbi = summary_for_powerbi.astype({
    'Month_Year': 'string',  # Or 'datetime64[ns]' if keeping timestamp format
    'ORIGIN_AIRPORT_ID_origin': 'string',
    'DEST_AIRPORT_ID': 'string',
    'estimated_demand': 'float64',
    'optimized_demand': 'float64',
    'price_proxy': 'float64',
    'original_revenue': 'float64',
    'optimized_revenue': 'float64',
    'revenue_uplift_percent': 'float64',
    'flight_count': 'int64',
    'cluster': 'int64'
})

# Export to CSV
summary_for_powerbi.to_csv('C:/Users/tejas/OneDrive/Desktop/flight dataset/route_summary_powerbi.csv', index=False)

# Export final dataset
route_data.to_csv('C:/Users/tejas/OneDrive/Desktop/flight dataset/optimized_routes_corrected.csv', index=False)

# Visualize
plt.figure(figsize=(10, 5))
yearly_demand = route_data.groupby('year')['estimated_demand'].sum().reset_index()
sns.lineplot(x='year', y='estimated_demand', data=yearly_demand, marker='o')
plt.title('Yearly Estimated Demand')
plt.xlabel('Year')
plt.ylabel('Demand')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
sns.scatterplot(x=route_data['flight_count'], y=route_data['optimized_demand'], hue=route_data['cluster'])
plt.xlabel('Number of Flights')
plt.ylabel('Optimized Demand')
plt.title('Price Elasticity vs Flight Count')
plt.show()

pivot = route_data.groupby(['ORIGIN_AIRPORT_ID_origin', 'DEST_AIRPORT_ID'])['estimated_demand'].sum().unstack(fill_value=0)
sns.heatmap(pivot, cmap='coolwarm')
plt.title('Route Performance Heatmap')
plt.show()
