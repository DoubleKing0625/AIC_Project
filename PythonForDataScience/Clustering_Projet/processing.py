from sklearn.mixture import GMM
from sklearn import cluster
import numpy as np

def gmm_segmentation(n_component=None, X=None, data=None):
    model = GMM(n_components=n_component)
    model.fit(X)

    classes = model.predict(X)
    res_X = X.copy()
    res_X["classes"] = classes
    res_data = data.copy()
    res_data["classes"] = classes
    print(res_data.groupby('classes').mean())



def k_means(n_clusters=None, X=None, data=None, init_centers=None, max_workers=-1):
    is_dataset_big_enough = len(X.index) > 1
    n_clusters_check = len(X.index) > n_clusters
    if is_dataset_big_enough and n_clusters_check:

        if init_centers:
            model = cluster.KMeans(n_clusters=n_clusters, init=init_centers, precompute_distances='auto', n_jobs=max_workers)
        else:
            model = cluster.KMeans(n_clusters=n_clusters, init='k-means++', precompute_distances='auto', n_jobs=max_workers)

        model.fit(X)
        classes = model.labels_
        centers = model.cluster_centers_

    else:
        classes = [0] * len(X.index)
        centers = X.iloc[0]

    res_X = X.copy()
    res_X["classes"] = classes
    res_data = data.copy()
    res_data["classes"] = classes
    print(res_data.groupby('classes').mean())
    return {'classes': classes, 'centers': centers}


def gmm_components(X=None, threshold=None):
    n_components = [1, 2, 3, 4, 5]
    metric = []

    for n_component in n_components:

        model = GMM(n_components=n_component)
        model.fit(X)

        classes = list(model.predict(X))
        classes_count = [classes.count(i) for i in range(n_component)]
        classes_vol = [threshold > i for i in classes_count]

        aic_score = model.aic(X)

        if classes_vol.count(True) > 0:
            aic_score = + np.inf

        metric.append(aic_score)

        print('GMM with ' + str(n_component) + ': aic ' + str(metric[n_component - 1]))
        
        for mean, std, cat in zip(model.means_, np.sqrt(model.covars_), range(n_component)):
            ic1 = mean[0] - (1.96 * std[0]) / classes.count(cat)
            ic2 = mean[0] + (1.96 * std[0]) / classes.count(cat)
            print ('confidence interval for class:  ' + str(cat + 1) + ': ' + str(ic1) + str(ic2) + ' with ' + str(classes.count(cat)) + ' transactions ' + ' while threshold is ' + str(threshold))

    idx_min = metric.index(min(metric)) + 1

    return idx_min