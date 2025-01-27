#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Import required libraries
import pandas as pd
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("C:/Users/tejas/Downloads/flight dataset/Clean_Dataset.csv")

# Inspect the dataset
print("Dataset Head:")
print(data.head())

print("\nDataset Info:")
print(data.info())

print("\nMissing Values:")
print(data.isnull().sum())

# Ensure numerical columns are properly formatted
data['price'] = pd.to_numeric(data['price'], errors='coerce')
data['demand'] = pd.to_numeric(data['duration'], errors='coerce')  # Simulating 'Demand' as 'duration'
data['capacity'] = pd.to_numeric(data['days_left'], errors='coerce')  # Simulating 'Capacity' as 'days_left'

# Drop rows with any invalid numerical values
data = data.dropna()

# Extract relevant columns for optimization
prices = data['price'].values  # Prices per flight
demands = data['demand'].values  # Using 'duration' as a proxy for demand
capacities = data['capacity'].values  # Using 'days_left' as a proxy for capacity

# Define the number of flights (decision variables)
num_flights = len(prices)

# Objective function: maximize revenue (minimizing negative revenue for linprog)
c = -1 * prices

# Constraints
# Capacity constraint: flight allocation <= capacity
A_ub = np.eye(num_flights)  # Each flight must not exceed capacity
b_ub = capacities

# Demand constraint: flight allocation must meet demand
A_eq = np.eye(num_flights)  # Each flight should meet demand
b_eq = demands

# Bounds for decision variables: 0 to capacity
x_bounds = [(0, capacity) for capacity in capacities]

# Solve the optimization problem
result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=x_bounds, method='highs')

# Check the result
if result.success:
    print("\nOptimization Successful!")
    print("Optimal Flight Allocations (number of seats per flight):")
    print(result.x)
    print("Maximum Revenue Achieved: $", -1 * result.fun)
else:
    print("\nOptimization Failed!")
    print(result.message)

# Visualization: Revenue by Flight
optimal_allocations = result.x
revenue_by_flight = optimal_allocations * prices

plt.figure(figsize=(10, 6))
plt.bar(range(num_flights), revenue_by_flight, color='skyblue')
plt.xlabel('Flight Index')
plt.ylabel('Revenue ($)')
plt.title('Revenue per Flight')
plt.show()


# In[ ]:


# Import required libraries
import pandas as pd
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import eye
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("C:/Users/tejas/Downloads/flight dataset/Clean_Dataset.csv")

# Extract relevant columns
prices = data['price'].values  # Prices per flight
demands = data['duration'].values  # Using 'duration' as a proxy for demand
capacities = data['days_left'].values  # Using 'days_left' as a proxy for capacity

# Ensure demands do not exceed capacities
demands = np.minimum(demands, capacities)

# Number of flights
num_flights = len(prices)

# Objective function: maximize revenue (minimize negative revenue)
c = -1 * prices

# Upper bound constraints: flight allocation <= capacity
A_ub = eye(num_flights, format='csr')  # Sparse identity matrix
b_ub = capacities

# Relaxed demand constraint: flight allocation >= demand
A_eq = None  # No strict equality constraints
b_eq = None

# Bounds for decision variables: 0 to capacity
x_bounds = [(0, capacity) for capacity in capacities]

# Solve the optimization problem
result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=x_bounds, method='highs')

# Check the result
if result.success:
    print("\nOptimization Successful!")
    print("Optimal Flight Allocations (number of seats per flight):")
    print(result.x)
    print("Maximum Revenue Achieved: $", -1 * result.fun)
    
    # Visualization: Revenue by Flight
    optimal_allocations = result.x
    revenue_by_flight = optimal_allocations * prices

    plt.figure(figsize=(10, 6))
    plt.bar(range(num_flights), revenue_by_flight, color='skyblue')
    plt.xlabel('Flight Index')
    plt.ylabel('Revenue ($)')
    plt.title('Revenue per Flight')
    plt.show()
else:
    print("\nOptimization Failed!")
    print(result.message)

# Debugging: Analyze the problem
if not result.success:
    print("\nAnalyzing Infeasibility...")
    print("Max Demand:", demands.max())
    print("Max Capacity:", capacities.max())
    print("Demands Exceeding Capacities:", sum(demands > capacities))

