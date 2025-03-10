import numpy as np
from sklearn.cluster import kmeans_plusplus as kmeans_plusplus_v1
from kmeans_plusplus import _kmeans_plusplus, _kmeans_plusplus_v2
import numpy as np
from sklearn import cluster
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.metrics.pairwise import euclidean_distances


class Big_KMeans:
    def __init__(self, n_clusters=3, tolerance=0.01, max_iter=100, runs=1, random_state=None):
        self.n_clusters = n_clusters
        self.tolerance = tolerance
        self.cluster_means = np.zeros(n_clusters)
        self.max_iter = max_iter
        self.runs = runs
        self.random_state = random_state

    def fit(self, X):
        row_count, col_count = X.shape
        X_labels = np.zeros(row_count)

        costs = np.zeros(self.runs)
        all_clusterings = []

        for i in range(self.runs):
            cluster_means = self.__initialize_means(X, self.n_clusters, self.random_state)

            for _ in range(self.max_iter):
                previous_means = np.copy(cluster_means)

                distances = self.__compute_distances(X, cluster_means, row_count)

                X_labels = self.__label_examples(distances)

                cluster_means = self.__compute_means(X, X_labels, col_count)

                clusters_not_changed = np.abs(cluster_means - previous_means) < self.tolerance
                if np.all(clusters_not_changed) != False:
                    break

            # X_with_labels = np.append(X, X_labels[:, np.newaxis], axis=1)
            all_clusterings.append((cluster_means, X_labels))
            costs[i] = self.__compute_cost(X, X_labels, cluster_means)

        best_clustering_index = costs.argmin()

        self.cost_ = costs[best_clustering_index]

        return all_clusterings[best_clustering_index]

    def fit_merge(self, X, y):
        y_cls = np.unique(y)
        n_clusters_unknown = self.n_clusters - len(np.unique(y))
        X_src = X[:len(y), ]
        X = X[len(y):, ]
        row_count, col_count = X.shape
        cluster_means = np.zeros((self.n_clusters, col_count))
        given_centers = []
        for c in y_cls:
            cluster_means[c,:] = np.mean(X_src[y==c,:], axis=0)
            given_centers.append(cluster_means[c,:])
        diff_cls = np.array([c for c in range(self.n_clusters) if c not in y_cls])
        given_centers = np.stack(given_centers, axis=0)
        X_labels = np.zeros(row_count)

        costs = np.zeros(self.runs)
        all_clusterings = []

        for i in range(self.runs):
            cluster_means[diff_cls, :] = self.__initialize_means(X, n_clusters_unknown, given_centers, 200,
                                                                self.random_state)
            # cluster_means[diff_cls, :] = self.__initialize_means(X, n_clusters_unknown, given_centers, self.n_clusters,
            #                                                      self.random_state)
            for _ in range(self.max_iter):
                previous_means = np.copy(cluster_means)

                distances = self.__compute_distances(X, cluster_means, row_count)

                X_labels = self.__label_examples(distances)

                cluster_means = self.__compute_means_merge(X, X_labels, col_count, X_src, y)

                clusters_not_changed = np.abs(cluster_means - previous_means) < self.tolerance
                if np.all(clusters_not_changed) != False:
                    break

            # X_with_labels = np.append(X, X_labels[:, np.newaxis], axis=1)
            all_clusterings.append((cluster_means, X_labels))
            costs[i] = self.__compute_cost(X, X_labels, cluster_means)

        best_clustering_index = costs.argmin()

        self.cost_ = costs[best_clustering_index]
        return all_clusterings[best_clustering_index]

    def __post_initialize(self, centers, given_centers, k, indices):
        dist = euclidean_distances(centers, given_centers)
        dist_min = np.min(dist, axis=1)
        dist_argsort = np.argsort(dist_min)[::-1]
        index = dist_argsort[:k]
        return centers[index], indices[index]

    def __initialize_means(self, X, k, given_centers=None, max_k=None, random_state=None):
        if given_centers is not None:
            if max_k is not None:
                #centers, indices = kmeans_plusplus_v1(X, n_clusters=max_k, random_state=random_state)
                centers, indices = _kmeans_plusplus_v2(X, n_clusters=max_k, given_centers=given_centers, random_state=random_state)
                centers, indices = self.__post_initialize(centers, given_centers, k, indices)
            else:
                centers, indices = _kmeans_plusplus_v2(X, n_clusters=k, given_centers=given_centers, random_state=random_state)

        else:
            centers, indices = kmeans_plusplus_v1(X, n_clusters=k, random_state=random_state)
        return np.array(centers)

    def __compute_distances(self, X, cluster_means, row_count):
        distances = np.zeros((row_count, self.n_clusters))
        for cluster_mean_index, cluster_mean in enumerate(cluster_means):
            distances[:, cluster_mean_index] = np.linalg.norm(X - cluster_mean, axis=1)
        return distances

    def __label_examples(self, distances):
        return distances.argmin(axis=1)

    def __compute_means(self, X, labels, col_count):
        cluster_means = np.zeros((self.n_clusters, col_count))
        for cluster_mean_index, _ in enumerate(cluster_means):
            cluster_elements = X[labels == cluster_mean_index]
            if len(cluster_elements):
                cluster_means[cluster_mean_index, :] = cluster_elements.mean(axis=0)

        return cluster_means

    def __compute_means_merge(self, X, labels, col_count, X_src, y):
        cluster_means = np.zeros((self.n_clusters, col_count))
        for cluster_mean_index, _ in enumerate(cluster_means):
            if cluster_mean_index in np.unique(y):
                cluster_elements = np.concatenate([X[labels == cluster_mean_index], X_src[y == cluster_mean_index]], axis=0)
                cluster_means[cluster_mean_index, :] = cluster_elements.mean(axis=0)
            else:
                cluster_elements = X[labels == cluster_mean_index]
                if len(cluster_elements):
                    cluster_means[cluster_mean_index, :] = cluster_elements.mean(axis=0)

        return cluster_means


    def __compute_cost(self, X, labels, cluster_means):
        cost = 0
        for cluster_mean_index, cluster_mean in enumerate(cluster_means):
            cluster_elements = X[labels == cluster_mean_index]
            cost += np.linalg.norm(cluster_elements - cluster_mean, axis=1).sum()

        return cost





if __name__ == '__main__':
    # X = np.array([[1, 2], [1, 4], [1, 0], [5,6],[7,9],[10, 5],
    #               [10, 2], [10, 4], [10, 0], [9,12]])
    # kmeans = cluster.KMeans(n_clusters=2, n_init=1, random_state=0).fit(X)
    # print(kmeans.labels_)
    # print(kmeans.cluster_centers_)
    # kmeans2 = KMeans(n_clusters=2, random_state=0).fit(X)
    # print(kmeans2)

    num_components = 30
    num_components_cn = 10
    num_components_src = 20
    num_components_tar = num_components_src - num_components_cn + 1
    random_state = 12
    rng = np.random.RandomState(random_state)
    # Means, standard deviations, and weights for each component
    dim = 32
    # means = rng.rand(num_components,dim)
    means = rng.multivariate_normal(np.zeros(dim), np.eye(dim), num_components)
    ccov = np.absolute(rng.rand(num_components, dim))
    cov = np.eye(dim)
    # num_samples_src = 1800
    # num_samples_tar = 500
    num_samples_src = 500
    num_samples_tar = 1800
    tar_weights = rng.dirichlet([1 for i in range(num_components_tar, num_components)])
    src_label = rng.randint(num_components_src, size=num_samples_src)
    # print(src_label)
    # Initialize an empty array to store the generated data
    src_data = np.zeros((num_samples_src, dim))
    tar_data = np.zeros((num_samples_tar, dim))
    tar_label = rng.choice(np.arange(num_components_tar, num_components), size=num_samples_tar, p=tar_weights)
    print(tar_label)
    # Generate data points
    for i in range(num_samples_src):
        sl = src_label[i]
        # Generate a random sample from the selected component
        src_data[i, :] = rng.multivariate_normal(means[sl], cov * ccov[sl])
    for i in range(num_samples_tar):
        tl = tar_label[i]
        # Generate a random sample from the selected component
        tar_data[i, :] = rng.multivariate_normal(means[tl], cov * ccov[tl])

    data = np.concatenate([src_data, tar_data], axis=0)

    _, tar_pred = Big_KMeans(n_clusters=100, runs=1, random_state=0).fit_merge(data, src_label)

    unique, count = np.unique(tar_pred, return_counts=True)
    print(tar_pred.shape, len(unique), count, count.sum())
    print('nmi is', nmi(tar_pred, tar_label))
    tgt_member_new = tar_pred
    tar_label_known = tar_label[tar_label < 20]
    tgt_member_new_known = tgt_member_new[tar_label < 20]
    tar_label_unknown = tar_label[tar_label > 19]
    tgt_member_new_unknown = tgt_member_new[tar_label > 19]
    print('known acc', np.sum(tar_label_known == tgt_member_new_known) / np.sum(tar_label < 20), len(tar_label_known))
    # print('unknown nmi', nmi(tar_label_unknown, tgt_member_new_unknown), tar_label_unknown, tgt_member_new_unknown, len(tar_label_unknown))
    # define unknown as positive
    TP = np.sum((tar_label > 19) & (tgt_member_new > 19))
    TN = np.sum((tar_label < 20) & (tgt_member_new < 20))
    FP = np.sum((tar_label < 20) & (tgt_member_new > 19))
    FN = np.sum((tar_label > 19) & (tgt_member_new < 20))
    print('recall is ', TP / (TP + FN), TP, TP + FN)
    print('precision is', TP / (TP + FP), TP, TP + FP)

