"""
Clustering by measuring local Direction Centrality (CDC).
"""
import math
import numpy as np
from scipy.special import gamma
from scipy.spatial import ConvexHull
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array

def cdc_cluster(X, n_neighbors=20, ratio=0.9):
    """
    Perform CDC clustering from vector array or distance matrix.
    
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training instances.
        
    n_neighbors : int, default=20
        Number of nearest neighbors to consider.
        
    ratio : float, default=0.9
        Ratio for determining the DCM threshold. Must be between 0 and 1.
        
    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster labels for each point. Noisy samples are given the label -1.
    """
    X = check_array(X)
    
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be greater than 0")
    if not (0 < ratio < 1):
        raise ValueError("ratio must be between 0 and 1")

    k_num = n_neighbors
    num, d = X.shape

    # Nearest Neighbors
    # Note: We need k_num + 1 because the point itself is included
    nbrs = NearestNeighbors(n_neighbors=k_num+1, algorithm='ball_tree').fit(X)
    indices = nbrs.kneighbors(X, return_distance=False)
    # Exclude the point itself (first column)
    get_knn = indices[:, 1:k_num+1]

    angle_var = np.zeros(num)
    
    # Calculate DCM (Direction Centrality Metric)
    if d == 2:
        angle = np.zeros((num, k_num))
        for i in range(num):
            for j in range(k_num):
                delta_x = X[get_knn[i, j], 0] - X[i, 0]
                delta_y = X[get_knn[i, j], 1] - X[i, 1]
                if delta_x == 0:
                    if delta_y == 0:
                        angle[i, j] = 0
                    elif delta_y > 0:
                        angle[i, j] = math.pi / 2
                    else:
                        angle[i, j] = 3 * math.pi / 2
                elif delta_x > 0:
                    if math.atan(delta_y / delta_x) >= 0:
                        angle[i, j] = math.atan(delta_y / delta_x)
                    else:
                        angle[i, j] = 2 * math.pi + math.atan(delta_y / delta_x)
                else:
                    angle[i, j] = math.pi + math.atan(delta_y / delta_x)

        for i in range(num):
            angle_order = sorted(angle[i, :])

            for j in range(k_num - 1):
                point_angle = angle_order[j + 1] - angle_order[j]
                angle_var[i] = angle_var[i] + pow(point_angle - 2 * math.pi / k_num, 2)

            point_angle = angle_order[0] - angle_order[k_num - 1] + 2 * math.pi
            angle_var[i] = angle_var[i] + pow(point_angle - 2 * math.pi / k_num, 2)
            angle_var[i] = angle_var[i] / k_num

        angle_var = angle_var / ((k_num - 1) * 4 * pow(math.pi, 2) / pow(k_num, 2))
    else:
        for i in range(num):
            try:
                dif_x = X[get_knn[i], :] - X[i, :]
                cov = np.dot(dif_x, dif_x.T)
                if np.all(cov == 0):
                    map_x = dif_x
                else:
                    map_x = np.linalg.inv(np.diag(np.sqrt(np.diag(cov)))) @ dif_x
                
                hull = ConvexHull(map_x)
                simplex_num = len(hull.simplices)
                simplex_vol = np.zeros(simplex_num)

                for j in range(simplex_num):
                    simplex_coord = map_x[hull.simplices[j], :]
                    simplex_vol[j] = np.sqrt(max(0, np.linalg.det(np.dot(simplex_coord, simplex_coord.T)))) / gamma(d-1)

                angle_var[i] = np.var(simplex_vol)

            except Exception:
                angle_var[i] = 1

    # Determine threshold
    sort_dcm = sorted(angle_var)
    idx = math.ceil(num * ratio)
    if idx >= num:
        idx = num - 1
    T_DCM = sort_dcm[idx]
    
    ind = np.zeros(num)
    for i in range(num):
        if angle_var[i] < T_DCM:
            ind[i] = 1 # Internal point

    near_dis = np.zeros(num)
    for i in range(num):
        knn_ind = ind[get_knn[i, :]]
        if ind[i] == 1: # Internal
            if 0 in knn_ind: # Has boundary neighbor
                bdpts_ind = np.where(knn_ind == 0)[0]
                bd_id = get_knn[i, bdpts_ind[0]]
                near_dis[i] = math.sqrt(np.sum((X[i, :] - X[bd_id, :])**2))
            else:
                near_dis[i] = float("inf")
                for j in range(num):
                    if ind[j] == 0:
                        temp_dis = math.sqrt(np.sum((X[i, :] - X[j, :])**2))
                        if temp_dis < near_dis[i]:
                            near_dis[i] = temp_dis
        else: # Boundary
            if 1 in knn_ind: # Has internal neighbor
                bdpts_ind = np.where(knn_ind == 1)[0]
                bd_id = get_knn[i, bdpts_ind[0]]
                near_dis[i] = bd_id # Storing ID of nearest internal point
            else:
                mark_dis = float("inf")
                for j in range(num):
                    if ind[j] == 1:
                        temp_dis = math.sqrt(np.sum((X[i, :] - X[j, :])**2))
                        if temp_dis < mark_dis:
                            mark_dis = temp_dis
                            near_dis[i] = j

    # Clustering
    cluster = np.zeros(num)
    mark = 1
    for i in range(num):
        if ind[i] == 1 and cluster[i] == 0:
            cluster[i] = mark
            for j in range(num):
                # Connectivity check
                if ind[j] == 1:
                    dist = math.sqrt(np.sum((X[i, :] - X[j, :])**2))
                    if dist <= near_dis[i] + near_dis[j]:
                        if cluster[j] == 0:
                            cluster[j] = cluster[i]
                        else:
                            # Merge clusters
                            temp_cluster = cluster[j]
                            temp_ind = np.where(cluster == temp_cluster)
                            cluster[temp_ind] = cluster[i]
            mark = mark + 1

    # Assign boundary points
    for i in range(num):
        if ind[i] == 0:
            cluster[i] = cluster[int(near_dis[i])]

    # Remap labels: start from 0, use -1 for unassigned (noise)
    # Original logic: 0 is unassigned, valid clusters >= 1
    
    unique_labels = np.unique(cluster)
    mapped_labels = np.full(num, -1, dtype=int)
    
    current_label = 0
    # Sort labels to ensure deterministic mapping (ignore 0)
    sorted_labels = sorted([l for l in unique_labels if l != 0])
    
    for old_label in sorted_labels:
        mapped_labels[cluster == old_label] = current_label
        current_label += 1
        
    return mapped_labels

class CDC(BaseEstimator, ClusterMixin):
    """
    Clustering by measuring local Direction Centrality (CDC).

    Parameters
    ----------
    n_neighbors : int, default=20
        Number of nearest neighbors to consider.
    
    ratio : float, default=0.9
        Ratio for determining the DCM threshold. Must be between 0 and 1.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point. Noisy samples are given the label -1.
    
    n_features_in_ : int
        Number of features seen during :term:`fit`.
        
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    References
    ----------
    Peng, D., Gui, Z.*, Wang, D. et al. Clustering by measuring local 
    direction centrality for data with heterogeneous density and weak connectivity. 
    Nat. Commun. 13, 5455 (2022). https://www.nature.com/articles/s41467-022-33136-9
    """
    def __init__(self, n_neighbors=20, ratio=0.9):
        self.n_neighbors = n_neighbors
        self.ratio = ratio

    def fit(self, X, y=None):
        """Compute CDC clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        self.labels_ = cdc_cluster(X, n_neighbors=self.n_neighbors, ratio=self.ratio)
        return self

    def fit_predict(self, X, y=None):
        """Compute clusters and return cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        self.fit(X)
        return self.labels_
