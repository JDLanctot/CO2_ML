import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Tuple, List

from predictionrl.visualize import plot_inertias, plot_silhouettes, plot_cluster_histogram, plot_scatter_columns, plot_scatter_clustered_columns

__all__ = ['apply_kmeans_clustering', 'find_optimal_clusters', 'name_clusters', 'run_clustering']

def apply_kmeans_clustering(df: pd.DataFrame, n_clusters: int, colname1: str, colname2: str, filter: str = "") -> Tuple:
    # FILTER THE DATA
    if filter != "":
        df = df[df['Gender'] == filter]

    # LET'S KEEP ONLY THE DATA WE NEED
    X = df[[colname1,colname2]].values

    # KMeans CLUSTERING
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)

    # LET'S APPEND THE CLUSTERS BACK TO THE DATAFRAME
    df['Cluster'] = kmeans.labels_

    return df, kmeans

def find_optimal_clusters(df: pd.DataFrame, colname1: str, colname2: str, filter: str = "") -> Tuple[List,List]:
    # FILTER THE DATA
    if filter != "":
        df = df[df['Gender'] == filter]

    # LET'S KEEP ONLY THE DATA WE NEED
    X = df[[colname1,colname2]].values

    # INIT OUR VARIABLES
    inertias = []
    silhouette_scores = []

    # TRY N CLUSTERS AND GET SCORES
    for i in range(2, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))

    plot_inertias(inertias)
    plot_silhouettes(silhouette_scores)

    return inertias, silhouette_scores

def name_clusters(df: pd.DataFrame, kmeans: KMeans, names: List) -> pd.DataFrame:
    if names != []:
        cluster_names = {i: f"{names[i]}" for i in range(kmeans.n_clusters)}
    else:
        cluster_names = {i: f"Cluster {i}" for i in range(kmeans.name_clusters)}

    df['Cluster Name'] = df['Cluster'].apply(lambda x: cluster_names[x])
    return df

def run_clustering(df: pd.DataFrame, colname1: str, colname2: str,
                   clusters: int = 4, visualize: bool = False) -> pd.DataFrame:
    if visualize:
        plot_scatter_columns(df, colname1, colname2)

    # QUESTION 1.3
    df_clusters, kmeans = apply_kmeans_clustering(df, clusters, colname1, colname2)

    # QUESTION 1.4
    inertias, silhouettes = find_optimal_clusters(df_clusters, colname1, colname2)

    # QUESTION 1.5
    df_named = name_clusters(df_clusters, kmeans, ["Low Income, Young",
                                                   "Middle Income, Young",
                                                   "Middle Income, Old",
                                                   "High Income"])

    if visualize:
        plot_cluster_histogram(df_named, kmeans)
        plot_scatter_clustered_columns(df_named, colname1, colname2)

    return df_named

