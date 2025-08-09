import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset and check first 5 rows
mall_data = pd.read_csv('data/Mall_Customers.csv')
print(mall_data.head())

# Check the shape
print("Shape", mall_data.shape)
# Check the column types
print('\nColumn types:\n', mall_data.dtypes)

# Get the missing values per column
print('\nMissing values per column')
print(mall_data.isna().sum())

X = mall_data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Set Income and Score on the same scale to avoid bias
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], alpha=0.6)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customers Income vs Spending Score')
plt.show()


