import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def analyze_online_retail_data(filename, n_clusters_range):
    # Load the dataset
    data = pd.read_csv(filename, encoding='ISO-8859-1', parse_dates=['InvoiceDate'], sep=',')
    print(data)
    print("Describe: " , data.describe())
    # Extract RFM information
    rfm_data = data.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (data['InvoiceDate'].max() - x.max()).days,
        'InvoiceNo': 'count',
        'Quantity': 'sum'
    })
    rfm_data.columns = ['Recency', 'Frequency', 'Monetary']

    # Apply standard scaling to RFM data
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_data)

    # Determine the optimal number of clusters using elbow method and silhouette score
    elbow_scores = []
    silhouette_scores = []
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(rfm_scaled)
        elbow_scores.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(rfm_scaled, kmeans.labels_))

    # Plot elbow method
    plt.plot(n_clusters_range, elbow_scores, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Elbow Score')
    plt.title('Elbow Method')
    plt.show()

    # Plot silhouette scores
    plt.plot(n_clusters_range, silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score')
    plt.show()

    # Determine the optimal number of clusters based on the highest silhouette score
    optimal_n_clusters = n_clusters_range[np.argmax(silhouette_scores)]
    print("Optimal number of clusters:", optimal_n_clusters)

    # Perform K-means clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    kmeans.fit(rfm_scaled)

    # Calculate cluster centers
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

    # Scatter plot of clusters with cluster centers
    plt.figure(figsize=(10, 6))
    plt.scatter(rfm_data['Recency'], rfm_data['Frequency'], c=kmeans.labels_, cmap='viridis')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', c='red', label='Cluster Centers')
    plt.xlabel('Recency')
    plt.ylabel('Frequency')
    plt.title('Cluster Analysis (2D)')
    plt.legend()
    plt.show()

    # Scatter plot of clusters with cluster centers
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot

    ax.scatter(rfm_data['Recency'], rfm_data['Frequency'], rfm_data['Monetary'], c=kmeans.labels_, cmap='viridis')
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], marker='x', c='red',
               label='Cluster Centers')
    ax.set_xlabel('Recency')
    ax.set_ylabel('Frequency')
    ax.set_zlabel('Monetary')
    ax.set_title('Cluster Analysis (3D)')
    ax.legend()
    plt.show()

    # Determine the distance of each data point from its cluster center
    distances = kmeans.transform(rfm_scaled)
    min_distances = np.min(distances, axis=1)

    # Set a threshold for outlier detection
    threshold = np.percentile(min_distances, 95)

    # Identify outliers
    outliers = rfm_data[min_distances > threshold]

    # Remove outliers from the data
    filtered_data = rfm_data[min_distances <= threshold]

    # Print outlier statistics
    n_outliers = len(outliers)
    print("Number of outliers:", n_outliers)
    print("Outliers:")
    print(outliers)

    # Return the filtered data (without outliers)
    return filtered_data


# Example usage of the function
filtered_data = analyze_online_retail_data("OnlineRetail.csv", range(2, 10))