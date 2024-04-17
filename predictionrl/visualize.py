import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from typing import List
from predictionrl.util import set_mpl

__all__ = ['plot_scatter_clustered_columns', 'plot_scatter_columns', 'plot_cluster_histogram'
           'plot_silhouettes', 'plot_inertias']

def plot_scatter_columns(df: pd.DataFrame, colname1: str, colname2: str, filter: str = "") -> None:
    # SET MY PERSONAL DEFAULT PREFERENCES FOR PLOTTING IN MPL
    set_mpl()

    # FILTER THE DATA
    if filter != "":
        df = df[df['Gender'] == filter]

    # PLOT
    plt.figure(figsize=(10, 6))
    plt.scatter(df[colname1], df[colname2])

    # PLOT FORMATTING
    plt.xlabel(colname1)
    plt.ylabel(colname2)
    plt.title(f'{colname1} vs {colname2}')
    plt.show()

def plot_inertias(inertias: List) -> None:
    set_mpl()
    plt.figure(figsize=(10, 5))
    plt.plot(range(2, 11), inertias, marker='o')
    plt.title('Inertia vs. Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.show()

def plot_silhouettes(silhouette_scores: List) -> None:
    set_mpl()
    plt.figure(figsize=(10, 5))
    plt.plot(range(2, 11), silhouette_scores, marker='o')
    plt.title('Silhouette Score vs. Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()

def plot_cluster_histogram(df: pd.DataFrame, kmeans: KMeans) -> None:
    set_mpl()

    plt.figure(figsize=(10, 6))
    plt.hist(df['Cluster Name'], bins=np.arange(kmeans.n_clusters + 1) - 0.5, edgecolor='black')
    plt.xticks(rotation=45)
    plt.title('Distribution of Clusters')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Customers')
    plt.show()

def plot_scatter_clustered_columns(df: pd.DataFrame, colname1: str, colname2: str) -> None:
    # SET MY PERSONAL DEFAULT PREFERENCES FOR PLOTTING IN MPL
    set_mpl()

    # PLOT
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df[colname1], df[colname2], c=df['Cluster'], cmap='viridis', edgecolor='k', s=50)

    # MAKE COLOUR MAP SO WE CAN REUSE IT
    cmap = plt.cm.viridis
    norm = mpl.colors.Normalize(vmin=df['Cluster'].min(), vmax=df['Cluster'].max())

    # PLOT FORMATTING
    plt.xlabel(colname1)
    plt.ylabel(colname2)
    plt.title(f'{colname1} vs {colname2} - Clustered')

    # CREATE LEGEND
    # LABELS FROM UNIQUE CLUSTER NAMES
    unique_clusters = df['Cluster'].unique()
    cluster_labels = df[['Cluster', 'Cluster Name']].drop_duplicates().sort_values('Cluster')['Cluster Name'].values

    # CREATE THE LEGEND HANDLES
    legend_handles = [plt.Line2D([], [], marker='o', color=cmap(norm(cluster)), linestyle='None',
                                 markersize=10, label=label) for cluster, label in zip(unique_clusters, cluster_labels)]

    plt.legend(handles=legend_handles, title="Clusters", loc='upper right')

    plt.show()


