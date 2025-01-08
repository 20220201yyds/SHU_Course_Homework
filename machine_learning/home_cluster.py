import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def distance(sample, centers):
    # 这里用差的平方来表示距离
    d = np.power(sample - centers, 2).sum(axis=1)
    cls = d.argmin()
    return cls


def clusters_show(clusters,centers,k):
    #plt.subplots()
    color = ["tomato", "slateblue","yellow","orange","green","purple","black"]
    plt.figure(figsize=(5, 5))
    plt.title("k: {}".format(k))
    plt.xlabel("Density", loc="center")
    plt.ylabel("Sugar Content", loc="center")
    # 用颜色区分k个簇的数据样本
    for i, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        plt.scatter(cluster[:, 0], cluster[:, 1], c=color[i],marker=".",s=100)
        plt.scatter(centers[:, 0], centers[:, 1], c=color[6], marker='.', s=150)


def k_means(samples, k):
    data_number = len(samples)
    centers_flag = np.zeros((k,))
    # 随机在数据中选择k个聚类中心
    centers = samples[np.random.choice(data_number, k, replace=False)]
    print('centers:::',centers)
    #step = 0
    while True:
        # 计算每个样本距离簇中心的距离, 然后分到距离最短的簇中心中
        clusters = [[] for i in range(k)]
        for sample in samples:
            ci = distance(sample, centers)
            clusters[ci].append(sample)

        for i, sub_clusters in enumerate(clusters):
            new_center = np.array(sub_clusters).mean(axis=0)

            if (centers[i] != new_center).all():
                centers[i] = new_center
            else:
                centers_flag[i] = 1
        
        if centers_flag.all():
            clusters_show(clusters, centers, k)
            break

    return centers

def split_data(samples, centers):

    # 根据中心样本得知簇数
    k = len(centers)
    clusters = [[] for i in range(k)]
    for sample in samples:
        ci = distance(sample, centers)
        clusters[ci].append(sample)

    return clusters

if __name__ == '__main__':

    data = np.array([
    [0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215],
    [0.403, 0.237], [0.481, 0.149], [0.437, 0.211], [0.666, 0.091], [0.243, 0.267],
    [0.245, 0.057], [0.343, 0.099], [0.639, 0.161], [0.657, 0.198], [0.360, 0.370],
    [0.593, 0.042], [0.719, 0.103], [0.359, 0.188], [0.339, 0.241], [0.282, 0.257],
    [0.748, 0.232], [0.714, 0.346], [0.483, 0.312], [0.478, 0.437], [0.525, 0.369],
    [0.751, 0.489], [0.532, 0.472], [0.473, 0.376], [0.725, 0.445], [0.446, 0.459]])
    centers = k_means(data, k=3)
    centers = k_means(data, k=4)
    centers = k_means(data, k=5)
    plt.show()
    clusters = split_data(data, centers=centers)
    print(clusters)