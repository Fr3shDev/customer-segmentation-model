import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
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

inertia = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)     # Sum of squared distances to cluster centers

plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters k')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()

kmeans = KMeans(n_clusters=5, random_state=1)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels back to original data for analysis
mall_data['Cluster'] = clusters

plt.figure(figsize=(8,6))

colors = ['red', 'blue', 'green', 'purple', 'orange']

for i in range(5):
    cluster_points = mall_data[mall_data['Cluster'] == i]
    plt.scatter(cluster_points['Annual Income (k$)'], cluster_points['Spending Score (1-100)'],s=50, color=colors[i], label=f'Cluster {i}')

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segments by K-Means Clustering')
plt.legend()
plt.show()

dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

mall_data['DBSCAN_Cluster'] = dbscan_labels

#Visualize DBSCAN results
plt.scatter(mall_data['Annual Income (k$)'], mall_data['Spending Score (1-100)'], c=mall_data['DBSCAN_Cluster'], cmap='viridis', s=50)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('DBSCAN Clustering')
plt.show()


