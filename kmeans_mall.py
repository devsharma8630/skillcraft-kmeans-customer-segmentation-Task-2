import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("Mall_Customers.csv")

# Select only needed columns
X = data.iloc[:, [3, 4]].values   # Annual Income & Spending Score

# Elbow Method to find best K
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Apply KMeans with K=5
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Plot Clusters
plt.scatter(X[y_kmeans==0,0], X[y_kmeans==0,1], s=100)
plt.scatter(X[y_kmeans==1,0], X[y_kmeans==1,1], s=100)
plt.scatter(X[y_kmeans==2,0], X[y_kmeans==2,1], s=100)
plt.scatter(X[y_kmeans==3,0], X[y_kmeans==3,1], s=100)
plt.scatter(X[y_kmeans==4,0], X[y_kmeans==4,1], s=100)

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300)
plt.title("Customer Segments")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()
