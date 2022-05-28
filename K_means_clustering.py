import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


X = pd.read_csv('Task2 - dataset - dog_breeds.csv', header = 'infer')

def initialise_centroids(X, k):
    Centroids = (X.sample(n=k))
    return Centroids
    
def compute_euclidean_distance(columns, Centroids):
    XD = X
    i=1
    for index1,row_c in Centroids.iterrows():
        ED=[]
        for index2,row_d in XD.iterrows():
            d1=(row_c["height"]-row_d["height"])**2
            d2=(row_c["tail length"]-row_d["tail length"])**2
            d3=(row_c["leg length"]-row_d["leg length"])**2
            d4=(row_c["nose circumference"]-row_d["nose circumference"])**2
            d=np.sqrt(d1+d2+d3+d4)
            ED.append(d)
        X[i]=ED
        i=i+1
    return ED

def kmeans(dataset, k):
    diff = 1
    j=0
    f = dataset.loc[:,["height", "tail length", "leg length", "nose circumference"]]
    Centroids = initialise_centroids(f, k)

    while(diff!=0):
        euclidean = compute_euclidean_distance(f, Centroids)

        C=[]
        for index,row in X.iterrows():
            min_dist=row[1]
            pos=1
            for i in range(k):
                if row[i+1] < min_dist:
                    min_dist = row[i+1]
                    pos=i+1
            C.append(pos)
        f["Cluster"]=C
        Centroids_new = f.groupby(["Cluster"]).mean()[["height", "tail length", "leg length", "nose circumference"]]
        if j == 0:
            diff=1
            j=j+1
        else:
            diff = (Centroids_new['height'] - Centroids['height']).sum() + (Centroids_new['tail length'] - Centroids['tail length']).sum() + (Centroids_new['leg length'] - Centroids["leg length"]).sum() + (Centroids_new['nose circumference'] - Centroids['nose circumference']).sum()
            print(diff.sum())
        Centroids = f.groupby(["Cluster"]).mean()[["height","tail length", "leg length", "nose circumference"]]
        
    color=['blue','green','cyan']
    for K in range(k):
        data=f[f["Cluster"]==K+1]
        plt.scatter(data["height"],data["tail length"],c=color[K])
    plt.scatter(Centroids["height"],Centroids["tail length"],c='red')
    plt.xlabel('Height')
    plt.ylabel('Tail Length')
    plt.show()
    for K in range(k):
        data=f[f["Cluster"]==K+1]
        plt.scatter(data["height"],data["leg length"])
    plt.scatter(Centroids["height"],Centroids["leg length"],c='red')
    plt.xlabel('Height')
    plt.ylabel('Leg Length')
    plt.show()

kmeans(X, 3)
kmeans(X, 2)