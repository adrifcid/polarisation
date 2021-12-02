### This script contains the functions used for clustering

import numpy as np

def agglomerative_clustering(y, method='ward', alpha=1, K=None, verbose=0):

    """
    Perform agglomerative clustering (of the given method) on given condensed distance matrix y. Adapted from scipy's 'linkage' method in https://github.com/scipy/scipy/blob/v1.7.0/scipy/cluster/hierarchy.py.
    Parameters
    ----------
    y : ndarray
        A condensed matrix stores the pairwise distances of the observations.
    method : str
        The linkage method: "single", "complete", "average", "centroid", "median",
        "ward", "weighted" or "polarisation".
    alpha : float
        Value of the alpha parameter, only used for computing polarisation and, if 
        method="polarisation", polarisation distance.
    K : float
        Value of the normalisation constant used for computing the polarisation (for any method, 
        although now the result is only correct for methods "centroid" and "polarisation").
        
    Returns
    -------
    Z : ndarray, shape (n - 1, 4)
        Computed linkage matrix.
    pol : ndarray, shape (n - 1,)
        Polarisation at each step, from beginning (point clusters) to next-to-last (2 clusters).
    """
    #y = _convert_to_double(np.asarray(y, order='c')) # to normalise input, I guess

    if y.ndim != 1:
        raise ValueError("`y` must be 1-dimensional.")

    if not np.all(np.isfinite(y)):
        raise ValueError("The condensed distance matrix must contain only "
                         "finite values.")
    L = y.shape[0]
    n = int(round((1+np.sqrt(1+8*L))/2))
    

  #  if method == 'single':
  #      result = _hierarchy.mst_single_linkage(y, n)      
  #  elif method in ['complete', 'average', 'weighted', 'ward']:
  #      result = _hierarchy.nn_chain(y, n, method_code)
  #  else:
  #      result = _hierarchy.fast_linkage(y, n, method_code)
  #  if method in ['complete', 'average', 'weighted', 'ward']:
  #      result = nn_chain(y, n, method)
    Z, pol = nn_chain(y, n, method, alpha, K, verbose)
    return Z, pol

def nn_chain(dists, n, method="ward", alpha=1, K=None, verbose=0):
    """Perform hierarchy clustering using nearest-neighbor chain algorithm. Adapted from cython function of the same name in
    https://github.com/scipy/scipy/blob/v1.7.0/scipy/cluster/_hierarchy.pyx.
    Parameters
    ----------
    dists : ndarray
        A condensed matrix stores the pairwise distances of the observations.
    n : int
        The number of observations.
    method : str
        The linkage method: "single", "complete", "average", "centroid", "median",
        "ward", "weighted" or "polarisation".
    alpha : float
        Value of the alpha parameter, only used for computing polarisation and, if 
        method="polarisation", polarisation distance.
    K : float
        Value of the normalisation constant used for computing the polarisation (for any method, 
        although now the result is only correct for methods "centroid" and "polarisation").
        
    Returns
    -------
    Z : ndarray, shape (n - 1, 4)
        Computed linkage matrix.
    pol : ndarray, shape (n - 1,)
        Polarisation at each step, from beginning (point clusters) to next-to-last (2 clusters).
    """
    ### DEFINE QUANTITIES
    Z_arr = np.empty((n - 1, 4))
    Z = Z_arr # this is just to rename the Z_arr: they are synchronised

    D = dists.copy()  # Distances between clusters.
    if method=="polarisation":
        D_centroids = dists.copy() #keep a parallel array with centroid distances

    # by default, set K to 1/max(pol) (for given population and with max dist = 1)
    if K == None:
        K = 2/n**(2+alpha) 
        
    size = np.ones(n, dtype=np.intc)  # Sizes of clusters.
    pol = np.empty(n-1) # polarisation at each level

    # Variables to store neighbors chain.
    cluster_chain = np.ndarray(n, dtype=np.intc)
    chain_length = 0

    ### ALGORITHM
    for k in range(n - 1):
        if (k < n-3) | (verbose == 0):
            print(f"Iteration {k}/{n-2}...", end='\r')
        else:
            print(f"Iteration {k}/{n-2}...") 
        #compute polarisation
        p=0
        if method=="polarisation":
            D_ctr = D_centroids
        else:
            D_ctr = D
        #    if k == 16:
        #        print(D_centroids)
        for i in range(n):
            ni = size[i]
            if ni == 0:
                continue    
            for j in range(n):
                nj = size[j]
                if nj == 0 or j==i:
                    continue       
                p += K*ni**(1+alpha)*nj*D_ctr[condensed_index(n, i, j)]
                if (k >= n-3) & (i > j) & (verbose > 0):
                    print(f"Cluster {i} (size {ni}) and cluster {j} (size {nj}) have dist {round(D_ctr[condensed_index(n, i, j)], 2)}")
        pol[k] = p
        if (k >= n-3) & (verbose > 0):
            print(f"and total polarisation is {(round(p, 2))}")
        
        #If chain emplty,pick one existing cluster (the first in size list)
        if chain_length == 0:
            chain_length = 1
            for i in range(n):
                if size[i] > 0:
                    cluster_chain[0] = i
                    break

        # Go through chain of neighbors until two mutual neighbors are found.
        while True:  #THIS IS EXITTED WITH THE BREAK BELOW
            x = cluster_chain[chain_length - 1]

            #CHECK FIRST DIST TO PREVIOUS CLUSTER IN CHAIN
            # We want to prefer the previous element in the chain as the
            # minimum, to avoid potentially going in cycles.
            if chain_length > 1:
                y = cluster_chain[chain_length - 2]
                current_min = D[condensed_index(n, x, y)]
            else:
                current_min = np.inf

            #FOR EVERY OTHER CLUSTER; CHECK DIST AND KEEP THE MIN
            for i in range(n):
                #SKIP DIST TO ITSELF AND TO ALREADY MERGED CLUSTERS
                if size[i] == 0 or x == i:
                    continue

                dist = D[condensed_index(n, x, i)]
                if dist < current_min:
                    current_min = dist
                    y = i

            #IF THE MIN DIST CLUSTER IS ALREADY IN CHAIN, IT IS THE PREVIOUS ELEMENT
            #AND WE HAVE MUTUAL MIN DIST, SO BREAK FROM WHILE AND MERGE CLUSTERS
            if chain_length > 1 and y == cluster_chain[chain_length - 2]:
                break
                
            #OTHERWISE; JUST CONTINUE STACKING CLUSTERS IN CHAIN
            cluster_chain[chain_length] = y
            chain_length += 1

        # Merge clusters x and y and pop them from stack.
        chain_length -= 2

        # This is a convention used in fastcluster.
        if x > y:
            x, y = y, x

        # get the original numbers of points in clusters x and y
        nx = size[x]
        ny = size[y]

        # Record the new node. Updating Z updates Z_arr as well
        Z[k, 0] = x
        Z[k, 1] = y
        Z[k, 2] = current_min
        Z[k, 3] = nx + ny
        size[x] = 0  # Cluster x will be dropped.
        size[y] = nx + ny  # Cluster y will be replaced with the new cluster

        # Update the distance matrix.
        #if polarisation method, update both pol. and centroid dists
        if method=="polarisation":
            for i in range(n):
                ni = size[i]
                if ni == 0 or i == y:
                    continue   
                cond_idx = condensed_index(n, i, y)
                D[cond_idx], D_centroids[cond_idx] = distance_update(
                    D_centroids[condensed_index(n, i, x)],
                    D_centroids[cond_idx],
                    D_centroids[condensed_index(n, x, y)], nx, ny, ni, method, alpha)
        else: 
            for i in range(n):
                ni = size[i]
                if ni == 0 or i == y:
                    continue
                D[condensed_index(n, i, y)] = distance_update(
                    D[condensed_index(n, i, x)],
                    D[condensed_index(n, i, y)],
                    current_min, nx, ny, ni, method)

    # Sort Z by cluster distances (the nn_chain algorithm does not produce that
    #order in general)
    order = np.argsort(Z_arr[:, 2], kind='mergesort')#an (n-1)x1 index array
    Z_arr = Z_arr[order]# orders rows of Z according to "order"
    pol = pol[order] #same for polarisation vector

    # Find correct cluster labels inplace (up to now, the new clusters had no label):
    # we label them accroding to their order of appearance in the newly sorted Z
    label(Z_arr, n)

    return Z_arr, pol


def distance_update(d_xi, d_yi, d_xy, size_x, size_y, size_i, method="ward", alpha=1):
    """
    A `linkage_distance_update` function calculates the distance from cluster i
    to the new cluster xy after merging cluster x and cluster y. Adapted from cython 
    function of the same name in
    https://github.com/scipy/scipy/blob/v1.7.0/scipy/cluster/_hierarchy_distance_update.pxi.
    
    Parameters
    ----------
    d_xi : double
        Distance from cluster x to cluster i
    d_yi : double
        Distance from cluster y to cluster i
    d_xy : double
        Distance from cluster x to cluster y
    size_x : int
        Size of cluster x
    size_y : int
        Size of cluster y
    size_i : int
        Size of cluster i
    method : str
        The linkage method: "single", "complete", "average", "centroid", "median",
        "ward", "weighted" or "polarisation".
    alpha : float
        Value of the alpha parameter, only used if method="polarisation".
    Returns
    -------
    d_xyi : double
        Distance from the new cluster xy to cluster i
    """
    if method=="single":
        return min(d_xi, d_yi)

    elif method=="complete":
        return max(d_xi, d_yi)

    elif method=="average":
        return (size_x * d_xi + size_y * d_yi) / (size_x + size_y)

    elif method=="centroid":
        return np.sqrt((((size_x * d_xi * d_xi) + (size_y * d_yi * d_yi)) -
                     (size_x * size_y * d_xy * d_xy) / (size_x + size_y)) /
                    (size_x + size_y))

    elif method=="median":
        return np.sqrt(0.5 * (d_xi * d_xi + d_yi * d_yi) - 0.25 * d_xy * d_xy)

    elif method=="ward":
        t = 1.0 / (size_x + size_y + size_i)
        return np.sqrt((size_i + size_x) * t * d_xi * d_xi +
                    (size_i + size_y) * t * d_yi * d_yi -
                    size_i * t * d_xy * d_xy)

    elif method=="weighted":
        return 0.5 * (d_xi + d_yi)
    
    elif method=="polarisation":
        #centroid dist from new cluster xy to i
        d_xy_i = np.sqrt((((size_x * d_xi * d_xi) + (size_y * d_yi * d_yi)) -
                     (size_x * size_y * d_xy * d_xy) / (size_x + size_y)) /
                    (size_x + size_y))
        d_pol_xy_i = d_xy_i*(size_i*(size_x+size_y)**(1+alpha)+
                       (size_x+size_y)*size_i**(1+alpha))
        return d_pol_xy_i, d_xy_i

def condensed_index(n, i, j):
    """
    Calculate the condensed index of element (i, j) in an n x n condensed
    matrix. Adapted from cython function of the same name in
    https://github.com/scipy/scipy/blob/v1.7.0/scipy/cluster/_hierarchy.pyx.
    """
    if i < j:
        return int(round(n * i - (i * (i + 1) / 2) + (j - i - 1)))
    elif i > j:
        return int(round(n * j - (j * (j + 1) / 2) + (i - j - 1)))
    
### The following class and function are just for relabeling clusters in the linkage matrix Z

class LinkageUnionFind:
    """Structure for fast cluster labeling in unsorted dendrogram. 
    Adapted from cython class of the same name in
    https://github.com/scipy/scipy/blob/v1.7.0/scipy/cluster/_hierarchy.pyx."""

    def __init__(self, n):
        self.parent = np.arange(2 * n - 1, dtype=np.intc)
        self.next_label = n
        self.size = np.ones(2 * n - 1, dtype=np.intc)

    def merge(self, x, y):
        self.parent[x] = self.next_label
        self.parent[y] = self.next_label
        size = self.size[x] + self.size[y]
        self.size[self.next_label] = size
        self.next_label += 1
        return size

    def find(self, x):
        p = x

        while self.parent[x] != x:
            x = self.parent[x]

        while self.parent[p] != x:
            p, self.parent[p] = self.parent[p], x

        return x
    
def label(Z, n):
    """Correctly label clusters in unsorted dendrogram.
    Adapted from cython function of the same name in
    https://github.com/scipy/scipy/blob/v1.7.0/scipy/cluster/_hierarchy.pyx."""
    uf = LinkageUnionFind(n)
    
    for i in range(n - 1):
        x, y = int(Z[i, 0]), int(Z[i, 1])
        x_root, y_root = uf.find(x), uf.find(y)
        if x_root < y_root:
            Z[i, 0], Z[i, 1] = x_root, y_root
        else:
            Z[i, 0], Z[i, 1] = y_root, x_root
        # weird: without this, new clusters are not correctly labelled
        #even if sizes are already assigned in nn__chain
        Z[i, 3] = uf.merge(x_root, y_root)