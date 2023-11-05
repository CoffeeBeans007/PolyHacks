import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import yfinance
import sys
sys.path.append('Data')

def return_writer(): 
    t = pd.read_csv("Data\tot_ret.csv",sep=";")
    t_date = t["Date"]
    t = t.iloc[:,1:]
    t= t.pct_change()
    print(t)
    

def macro_clustering(df, print_graphs = False) : 
    np.random.seed(100)
    # Elbow & gap stat calculs 
    inerties = []
    silhouettes = []
    K_range = range(2, 10)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)
        if k == 3:
            print(silhouette_score(df, kmeans.labels_))
        inerties.append(kmeans.inertia_) 
    
        silhouettes.append(silhouette_score(df, kmeans.labels_))

    if print_graphs == True : 
        # Elbow method graph 
        plt.figure(figsize=(8, 4))
        plt.plot(K_range, inerties, 'bo-')
        plt.xlabel('Nombre de clusters k')
        plt.ylabel('Inertie')
        plt.title('La méthode du coude pour le K-Means clustering')
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

df_macro = pd.read_csv("Data\Macro_datas.csv", sep=";")  
df_macro = df_macro.iloc[:60,1:]
# nombre de données macro à prendre est à determiner 

df_macro1 = df_macro[["Unempl_Rate","SP500_ret","Spread_10Y2Y","VIX"]]
df_macro1 = macro_clustering(df_macro1,True) 

df_macro2 = macro_clustering(df_macro,True) 
df_macro2 = df_macro2[["Unempl_Rate","SP500_ret","Spread_10Y2Y","VIX","cluster"]]
grouped1 = df_macro1.groupby('cluster')
grouped2 = df_macro2.groupby('cluster')
stats_by_cluster1 = [grouped1.mean(), grouped1.std()]
stats_by_cluster2 = [grouped2.mean(), grouped2.std()]

#print(df_macro["cluster"].value_counts())
print(stats_by_cluster1, stats_by_cluster2)

def reg_logistic(df, test_size = 0.2) :
    
    #seeding
    np.random.seed(100)
    
    facteurs = df.drop('cluster', axis=1) 
    df_clusters = df['cluster']  
    print(facteurs)
    
    # test and train split
    facteurs_train, facteurs_test, df_clusters_train, df_clusters_test = train_test_split(facteurs, df_clusters ,test_size, random_state=42)

    # Modelisation
    model = LogisticRegression(multi_class='ovr', max_iter=1000) 
    model.fit(facteurs_train, df_clusters_train)

    # probability prediction
    probabilities = model.predict_proba(facteurs_test)

    return(probabilities)  

# t = reg_logistic(df_macro)
# print(t)

