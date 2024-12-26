import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class Clustering:

    @staticmethod
    def elbow_method(df, features):
        X = df[features]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Experiment with values of k from 2 to 5 and use the elbow method to determine the optimal number
        inertia = []
        k_values = range(2, 6)

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertia.append(kmeans.inertia_)
        
        
        plt.figure(figsize=(8, 5))
        plt.plot(k_values, inertia, marker='o', linestyle='--', color='b')
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia (Sum of Squared Distances)")
        plt.title("Elbow Method for Optimal k")
        plt.grid(True)
        plt.show()

        return scaler, X_scaled

    @staticmethod
    def k_means_clustering(df, featurs, optimal_k, scaler, X_scaled):
        kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42)
        df['Cluster'] = kmeans_optimal.fit_predict(X_scaled)

        # Plot the clusters and describe each segment in terms of spending and income levels.
        plt.figure(figsize=(8, 5))
        for cluster in range(optimal_k):
            cluster_data = df[df['Cluster'] == cluster]
            plt.scatter(cluster_data['Longitude'], cluster_data['Latitude'], label=f"Cluster {cluster}")

        plt.scatter(kmeans_optimal.cluster_centers_[:, 0] * scaler.scale_[0] + scaler.mean_[0], 
                    kmeans_optimal.cluster_centers_[:, 1] * scaler.scale_[1] + scaler.mean_[1], 
                    c='red', marker='x', s=100, label="Centroids")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Segments (K-Means Clustering)")
        plt.legend()
        plt.grid(True)
        plt.show()





# X = df_copy[['Longitude', 'Latitude']]



# optimal_k = 4


