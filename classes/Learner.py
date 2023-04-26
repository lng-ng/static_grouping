import numpy as np
from kmodes.kmodes import KModes

from util import create_learning_matrix

#import sys
#np.set_printoptions(threshold=sys.maxsize)


class Learner:
    """
    Class representing the learning phase.

    learning_duration: int
        duration of the learning phase, in rounds
    """
    __slots__ = ['learning_duration']

    def __init__(self, learning_duration):
        self.learning_duration = learning_duration

    def batch_clustering(self, num_clusters, cluster_size_threshold,
                         all_timeseries, learning_start):
        """
        Groups the users using a clustering algorithm, according to their communication traffic during the learning phase.

        Parameters
        num_clusters : int
            intended number of clusters
        cluster_size_threshold: int
            the threshold at which the cluster is considered to be valid
        all_timeseries: (num_users, round_num) np.ndarray
            np.ndarray the entire timeseries of each user, acquired from the dataset
        learning_start : int
            the round at which learning phase begins

        Return
        cluster: cluster contain the members of the cluster
        cost: cost of clustering for elbow method
        learning_matrix: learning vectors of all users in the cluster
        """
        clusters = []
        learning_matrix = create_learning_matrix(all_timeseries[:, 0:learning_start+self.learning_duration],
                                                 self.learning_duration)
        print(learning_start+self.learning_duration)
        assert learning_matrix.shape == (len(all_timeseries), self.learning_duration)
        cost = -1
        print("Clustering started")
        # KShape
        """
        # _clusters = kshape(zscore(all_timeseries_transformed, axis=1), k)
        seed = 2
        X = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(learning_matrix)
        ks = KShape(n_clusters=num_clusters, n_init=10,max_iter=200,tol=1e-8,random_state=seed)
        labels = ks.fit_predict(X)
        inertia = ks.inertia_
        print(f"Inertia: {inertia}")
        """
        # KModes
        km = KModes(n_clusters=num_clusters, init='Huang', n_init=10, verbose=0)
        labels = km.fit_predict(learning_matrix)
        cost = km.cost_
        # Random clustering
        # labels = np.random.choice(num_clusters, learning_matrix.shape[0])
        _clusters = []
        for i in range(num_clusters):
            _clusters.append([None, np.where(labels == i)[0].tolist()])
        # Random clustering
        print("Clustering finished")
        print(f"Number of clusters: {len(_clusters)}")
        n = 1
        for centroid, cluster in _clusters:
            if len(cluster) < cluster_size_threshold:
                print(f"Invalid cluster found with size {len(cluster)}")
            if len(cluster) == 0:
                continue
            """
            #print(f"Cluster size: {len(cluster)}")
            #print(f"Cluster vectors:\n{cluster_timeseries}")

            online_timeseries = all_timeseries[cluster]
            #online_timeseries = online_timeseries[:,learning_start+self.learning_duration:]
            print(f"Cluster {n}")
            print(f"Schedule:\n{cluster_schedule}")
            print("User data:")
            for lv, ts in zip(cluster_timeseries, online_timeseries):
                online_start = learning_start + self.learning_duration
                print(f"Communication vector between arrival and before start of learning phase:"
                      f"\n{np.trim_zeros(ts[0:learning_start], 'f')}")
                print(f"Communication vector during learning phase:\n{ts[learning_start:online_start]}")
                print(f"Learning vector:\n{lv}")
                print(f"Communication vector during online phase:\n{ts[online_start:]}")
                #while start % 24 != 0:
                 #   start += 1
                #tmp = np.asarray(ov[start:-1])
              #  tmp = np.reshape(tmp, (-1, 24))
                print("---------")
            print("--------------------------------------------------------------")
            """
            clusters.append(cluster)
            n += 1
        return clusters, cost, learning_matrix
