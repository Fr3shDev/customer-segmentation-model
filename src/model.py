import pandas as pd

mall_data = pd.read_csv('data/Mall_Customers.csv')
print(mall_data.head())

# Check the shape
print("Shape", mall_data.shape)
# Check the column types
print('\nColumn types:\n', mall_data.dtypes)

# Get the missing values per column
print('\nMissing values per column')
print(mall_data.isna().sum())

print('\nNumeric summary statistics:')
print(mall_data.describe())

cleaned = mall_data.dropna(subset=['Annual Income', 'Spending Score'])
