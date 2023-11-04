import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt



def macro_clustering(df, print_graphs = False) : 
    
    # Elbow & gap stat calculs 
    inerties = []
    silhouettes = []
    K_range = range(1, 10)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)
        inerties.append(kmeans.inertia_)
        # 
        if k > 1:
            silhouettes.append(silhouette_score(df, kmeans.labels_))

    if print_graphs == True : 
        # Elbow method graph 
        plt.figure(figsize=(8, 4))
        plt.plot(K_range, inerties, 'bo-')
        plt.xlabel('Nombre de clusters k')
        plt.ylabel('Inertie')
        plt.title('La méthode du coude pour le K-Means clustering')
        plt.show()

        # Gap-stat graph 
        plt.figure(figsize=(8, 4))
        plt.plot(K_range[1:], silhouettes, 'go-')
        plt.xlabel('Nombre de clusters k')
        plt.ylabel('Score de Silhouette')
        plt.title('Le score de silhouette pour le K-Means clustering')
        plt.show()

    # Optimale value 
    k_optimal = 3 

    # Clustering kmeans
    kmeans_optimal = KMeans(n_clusters=k_optimal, random_state=42)
    kmeans_optimal.fit(df)
    clusters = kmeans_optimal.labels_

    # Ajouter les étiquettes des clusters au dataframe
    df['cluster'] = clusters

    # Afficher les résultats
    return df

df_macro = pd.read_csv('Data\Macro_datas.csv',sep=";")  
df_macro = df_macro.iloc[:48,1:]
# nombre de données macro à prendre est à determiner 

df_macro = macro_clustering(df_macro,True) 
clust1 = df_macro[df_macro["cluster"]==1]
print(clust1)


