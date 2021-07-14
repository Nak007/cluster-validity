'''
Available methods are the followings:
[1] dispersion
[2] calinski_harabasz, plot_ch
[3] elbow_index, plot_elbow
[4] silhouette, plot_silhouette
[5] wb_index, plot_wb
[6] gap_statistics, plot_gap
[7] hartigen_index, plot_hartigan

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 13-11-2020

'''

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

__all__ = ['dispersion', 
           'calinski_harabasz', 
           'elbow_index', 
           'silhouette', 
           'wb_index', 
           'gap_statistics', 
           'hartigen_index', 
           'plot_elbow', 
           'plot_wb', 
           'plot_ch',
           'plot_hartigan',
           'plot_silhouette',
           'plot_gap']

def dispersion(X, labels, metric='euclidean'):
    
    '''
    Determine the dispersion within cluster (SSW) 
    and between clusters (SSB), which are defined 
    as follows:
    
            SSW = ∑ (‖ x, C ‖^2 | x ∈ C)

            SSB = ∑ (‖ C, μ ‖^2 | C ∈ K)

    where ‖ x, C ‖ is a distance from instance to 
    its cluster. Such distance can be any metric. 
    K is a set of clusters {C1, C2, ..., Cn} from 
    X and μ is a centroid of X.
    
    .. versionadded:: 13-11-2020
      
    Parameters
    ----------
    X : 2d-array, shape of (n_sample, n_feature)
        Training instances. Each row corresponds
        to a single data point.
        
    labels : 2d-array, shape of (n_sample, n_cluster)
        Predicted labels of instances for respective 
        n_cluster or kth cluster.
    
    metric: str, default="euclidean"
        The distance metric to use. Passes any option 
        accepted by `scipy.spatial.distance.cdist`.
    
    Returns
    -------
    intra_disp : array-like, shape (n_cluster,)
        List of within-cluster dispersions wrt. kth
        cluster.
    
    extra_disp : array-like, shape (n_cluster,)
        List of between-clusters dispersions wrt. kth
        cluster. 
    
    k_clusters : array-like, shape (n_cluster,)
        Number of unique labels wrt. kth cluster.
        
    '''
    # Initialize loop parameters.
    intra_disp, extra_disp, k_clusters = [], [], []
    mean = np.mean(X, axis=0) # Center of X.
    
    # Main loop for distance.
    for k in range(labels.shape[1]):
        
        # Reset parameters.
        intra, extra = 0, 0
        
        # Determine i-th cluster centriod, (i ∈ k, and 
        # k is number of clusters) by calculating  mean 
        # values of features of instances that belong 
        # to i-th cluster.
        y = labels[:,k].copy()
        centrs = np.array([np.mean(X[y==c,:],axis=0).ravel() 
                           for c in np.unique(y) if c >= 0])
        
        # Compute distances between clusters and their
        # members (instances).
        for c in np.unique(y):
            # arguments for `scipy.spatial.distance.cdist`.
            args = (X[y==c,:], centrs[[c],:])
            # within-cluster dispersion.
            intra += (cdist(*args, metric=metric)**2).sum()
            # between-cluster dispersion.
            extra += len(args[0]) * np.sum((centrs[[c],:]-mean)**2)
        
        # Count number of clusters for `k` trial.
        unq_labels = [c for c in np.unique(y) if c>-1]
        k_clusters.append(len(unq_labels))
        
        # Store 'intra' and 'extra' displacements.
        intra_disp.append(intra)
        extra_disp.append(extra)
             
    return intra_disp, extra_disp, k_clusters

def calinski_harabasz(X, labels, metric='euclidean'):
    
    '''
    Compute the Calinski and Harabasz index (CH), which 
    is defined as follows:
    
             CH = [SSB.(N - K)]/[SSW.(K - 1)]
    
    where `SSW` is the within-group dispersion matrix 
    for data clustered into `K` clusters, while `SSB` 
    is a dispersion between clusters. `N` is a number
    of samples (`X`), and `K` is a number of clusters.
    
    .. versionadded:: 13-11-2020
    
    References
    ----------
    .. [1] https://www.researchgate.net/publication/
           257138057_An_examination_of_indices_for_
           determining_the_number_of_clusters_NbClust_
           Package
    .. [2] https://scikit-learn.org/stable/modules/
           generated/sklearn.metrics.calinski_harabasz_
           score.html
        
    Parameters
    ----------
    X : 2d-array, shape of (n_sample, n_feature)
        Training instances. Each row corresponds to a 
        single data point.
        
    labels : 2d-array, shape of (n_sample, n_cluster)
        Predicted labels of instances for respective 
        n_cluster or kth cluster.
    
    metric: str, default="euclidean"
        The distance metric to use. Passes any option 
        accepted by `scipy.spatial.distance.cdist`.
        
    Returns
    -------
    score : array-like, shape (n_cluster,)
        The resulting Calinski-Harabasz index. The 
        maximum value of the index is taken as indicating 
        the correct number of clusters in the data.
    
    k_clusters : array-like, shape (n_cluster,)
        List of number  of unique labels wrt. kth cluster.
    
    '''
    # The denominator takes maximum between 1, and 
    # `k_clusters` - 1 as to avoid the error when 
    # `k_clusters` equals to 1.
    inputs = np.vstack(dispersion(X, labels, metric)).T
    score = [1 if x[0] == 0 else (x[1]*(X.shape[0]-x[2]))
             /max((x[0]*(x[2]-1)),1) for x in inputs]
    k_clusters = inputs[:,2].astype(int).tolist()
    return score, k_clusters

def elbow_index(X, labels, metric='euclidean'):
    
    '''
    This method looks at change of within-group 
    dispersion (`SSW`). One should choose a number of 
    clusters so that adding another cluster doesn't 
    reduce significant amount of `SSW`. 
    
    More precisely, if one plots `SSW` by the clusters 
    against the number of clusters (`Elbow plot`), one 
    should choose the point before marginal gain (ΔSSW) 
    becomes minute or insignificant. Thus, calculating 
    2nd order of relative change in distance function 
    is introduced to help detect such point, which can 
    be expressed as follows:
    
                W = ∑ (‖ x, C ‖^2 | x ∈ C)
    
              γ(k) = (δ(k) - δ(k-1))/|δ(k-1)|
                   
              δ(k) = (W(k) - W(k-1))/|W(k-1)|
    
    where ‖ x, C ‖ is a distance from instance to its 
    cluster. Such distance can be any metric. k is a  
    set of clusters {C1, C2, ..., Cn} from X and Gamma 
    is the  rate of change. 
    
    .. versionadded:: 13-11-2020

    Parameters
    ----------
    X : 2d-array, shape of (n_sample, n_feature)
        Training instances. Each row corresponds to a 
        single data point.
        
    labels : 2d-array, shape of (n_sample, n_cluster)
        Predicted labels of instances for respective 
        n_cluster or kth cluster. The number of n_cluster 
        must be more than 3.
    
    metric: str, default="euclidean"
        The distance metric to use. Passes any option  
        accepted by `scipy.spatial.distance.cdist`.

    Returns
    -------
    Gamma : array-like, shape (n_cluster-2,)
        The resulting rate of change in distances. The  
        maximum value of the index is taken as indicating 
        the correct number of clusters in the data.
    
    k_clusters : array-like, shape (n_cluster-2,)
        List of number  of unique labels wrt. kth cluster.
        
    '''
    # 2nd derivative of the distances.
    inputs = np.vstack(dispersion(X, labels, metric)).T
    Delta = np.diff(inputs[:,0])/inputs[:-1,0]
    Gamma = np.diff(Delta)/abs(Delta[:-1])
    k_clusters = inputs[1:-1,2].astype(int).tolist()
    return Gamma, k_clusters

def silhouette(X, labels, metric='euclidean', 
               p_sample=0.5, random_state=None,):
    
    '''
    Silhouette coefficient is calculated using the mean 
    intra-cluster distance (a) and the mean nearest-
    cluster distance (b) for each sample. The formula 
    (for a sample) is expressed as follows: 
    
            Silhouette = (b - a) / max{a, b}
  
    where `a` is a mean intra-cluster, and `b` is a mean 
    nearest-cluster distance.
    
    .. versionadded:: 13-11-2020
    
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Silhouette_
           (clustering)
    .. [2] https://scikit-learn.org/stable/auto_examples/
           cluster/plot_kmeans_silhouette_analysis.html
        
    Parameters
    ----------
    X : 2d-array, shape of (n_sample, n_feature)
        Training instances. Each row corresponds to a 
        single data point.
        
    labels : 2d-array, shape of (n_sample, n_cluster)
        Predicted labels of instances for respective 
        n_cluster or kth cluster.
    
    metric: str, default="euclidean"
        The distance metric to use. Passes any option  
        accepted by `scipy.spatial.distance.cdist`.
    
    p_sample : float, default=0.5
        Percent of samples.
    
    random_state : int, defualt=None
        Seed for the random number generator.
    
    Returns
    -------
    score : array-like, shape (n_cluster,)
        The resulting silhouette coefficients. The  
        maximum value of the index is taken as indicating 
        the correct number of clusters in the data.

    k_clusters : array-like, shape (n_cluster,)
        List of number of unique labels wrt. kth cluster.
    
    '''
    # Keyword arguments for `silhouette_score`
    kwargs = dict(metric=metric, random_state=random_state, 
                  sample_size=int(X.shape[0]*p_sample))
    
    # When number of unique labels equals to 1, Silhouette 
    # coefficient defaults to 0 as nearest-cluster distance
    # cannot be determined.
    score, k_clusters = [], []
    for n in range(labels.shape[1]):
        k_clusters.append(np.unique(labels[:,n]).shape[0])
        if k_clusters[-1]>1:
            args = (X, labels[:,n])
            score.append(silhouette_score(*args,**kwargs))
        else: score.append(0)
    return score, k_clusters

def wb_index(X, labels, metric='euclidean'):
    
    '''
    Compute the WB-index (WB), which is defined as 
    follows:

                    WB = K.SSW/SSB
    
    where SSW, SSB, and K are a dispersion within 
    cluster, a dispersion between clusters, and number 
    of clusters, respectively.
  
    .. versionadded:: 13-11-2020
    
    Parameters
    ----------
    X : 2d-array, shape of (n_sample, n_feature)
        Training instances. Each row corresponds to a 
        single data point.
        
    labels : 2d-array, shape of (n_sample, n_cluster)
        Predicted labels of instances for respective 
        n_cluster or kth cluster.
    
    metric: str, default="euclidean"
        The distance metric to use. Passes any option  
        accepted by `scipy.spatial.distance.cdist`.
    
    Returns
    -------
    WB : array-like, shape (n_cluster,)
        The resulting WB indices in logarithmic form, 
        LOG(WB). The minimum value of the index is taken 
        as indicating the correct number of clusters in 
        the data.
    
    k_clusters : array-like, shape (n_cluster,)
        List of number of unique labels wrt. kth cluster.
        
    '''
    # The resulting WB indices will be
    inputs = np.vstack(dispersion(X, labels, metric)).T
    SSW = inputs[:,0].copy()
    SSB = np.where(inputs[:,1]==0,1,inputs[:,1])
    k_clusters = inputs[:,2].astype(int).copy()
    WB = np.log((SSW/SSB)*k_clusters)
    return WB, k_clusters

def gap_statistics(X, labels, metric='euclidean', p_sample=0.5, 
                   n_bootstrap=5, random_state=None):
    
    '''
    The idea of the Gap statistic is to compare the SSW 
    (dispersion) to its expectation under an appropriate 
    null reference distribution i.e. a random uniform
    distribution. It can be mathematically expressed as:
    
            Gap(k) = Log(E[SSW(k,n)]) - Log(SSW)
            
    where k and n respresent number of clusters and
    number of bootstrappings, respectively. This function 
    uses `sklearn.cluster.KMeans` to determine cluster 
    centroids from a set of randomly distributed datasets.
    
    The optimum number of clusters is the smallest value k 
    such that Gap(k) ≥ Gap(k+1) − s(k+1), where s(k) is a 
    factor that takes into account the standard deviation 
    of the Monte-Carlo replicates SSW.
    
    .. versionadded:: 13-11-2020
    
    References
    ----------
    .. [1] https://statweb.stanford.edu/~gwalther/gap
    .. [2] https://datasciencelab.wordpress.com/tag/
           gap-statistic/
    
    Parameters
    ----------
    X : 2d-array, shape of (n_sample, n_feature)
        Training instances. Each row corresponds to a 
        single data point.
        
    labels : 2d-array, shape of (n_sample, n_cluster)
        Predicted labels of instances for respective 
        n_cluster or kth cluster.
    
    metric: str, default="euclidean"
        The distance metric to use. Passes any option  
        accepted by `scipy.spatial.distance.cdist`.
    
    p_sample : float, default=0.5
        Percent of samples.
    
    n_bootstrap : int, default=5
        Number of bootstrappings for Gap-statistics.
    
    random_state : int, defualt=None
        Seed for the random number generator.
    
    Returns
    -------
    Wks : array-like, shape (n_cluster,)
        The resulting LOG(Wk) from X. 
    
    Wkbs : array-like, shape (n_cluster,)
        The resulting E[LOG(Wk)] from a set of random
        uniform distribution of X.
    
    sk : array-like, shape (n_cluster,)
        Standard deviation from a set of random uniform 
        distribution of X.
    
    k_clusters : array-like, shape (n_cluster,)
        List of number of unique labels wrt. kth cluster.
        
    '''
    # Function to compute within-cluster sum of squares
    def intra_dist(x, C, metric):
        def dist(x): 
            mu = lambda x : x.mean(axis=0).reshape(1,-1)
            #np.linalg.norm(x-mu(x))**2/(2*len(x)) 
            return (cdist(x,mu(x),metric=metric)**2).sum()/(2*len(x))
        return np.sum([dist(x[C==n]) for n in np.unique(C)])
    
    # Initialize parameters.
    np.random.seed(random_state)
    n_sample = int(p_sample*X.shape[0])
    Wks = np.zeros(labels.shape[1])
    Wkbs = np.zeros(labels.shape[1])
    sk = np.zeros(labels.shape[1])
    
    # Create 'n' bootstapped dataset(s) with random uniform 
    # distribution for all features in 'X'.
    kwargs = [{'low':a, 'high':b, 'size':n_sample*n_bootstrap} 
              for (a,b) in zip(np.min(X,axis=0), np.max(X,axis=0))]
    bsX = [np.random.uniform(**k).reshape(-1,1) for k in kwargs]
    bsX = np.split(np.hstack(bsX), n_bootstrap, axis=0)
     
    for k in range(labels.shape[1]):
        
        # Compute 'Wk' for respective kth clusters.
        Wks[k] = np.log(intra_dist(X, labels[:,k], metric))
        
        # Loop parameters.
        BWkbs = np.zeros(n_bootstrap)
        kwargs = {'n_clusters' : len(np.unique(labels[:,k])), 
                  'random_state' : random_state}
        
        # Main loop for bootstrapped datasets.
        for n in range(n_bootstrap):
            
            # Compute within-cluster sum of squares for all  
            # clusters given 'kth' clusters.
            args = (bsX[n], KMeans(**kwargs).fit_predict(bsX[n]))
            BWkbs[n] = np.log(intra_dist(*args, metric=metric))
    
        # Wk is the pooled within-cluster sum of squares around  
        # the cluster means from differenct bootstrapped Xs.
        Wkbs[k] = sum(BWkbs)/n_bootstrap
        
        # Compute standard deviation.
        sk[k] = np.sqrt(sum((BWkbs-Wkbs[k])**2)/n_bootstrap)
        sk[k] = sk[k]*np.sqrt(1+1/n_bootstrap)
    
    # Number of clusters as per kth cluster.
    k_clusters = np.array([np.unique(labels[:,n]).shape[0] 
                           for n in range(labels.shape[1])])
    
    return Wks, Wkbs, sk, k_clusters

def hartigen_index(X, labels, metric='euclidean'):
    
    '''
    Compute the Hartigen index (1975), which is defined 
    as follows:
    
        H(k) = (SSW(k)/SSW(k+1) - 1).(N - k - 1)

    where `SSW` is the within-group dispersion matrix 
    for data clustered into `k` clusters. `N` is a 
    number of samples, and `k` is a number of clusters.
    
    .. versionadded:: 13-11-2020
    
    Parameters
    ----------
    X : 2d-array, shape of (n_sample, n_feature)
        Training instances. Each row corresponds
        to a single data point
        
    labels : 2d-array, shape of (n_sample, n_cluster)
        Predicted labels of instances for respective 
        n_cluster or kth cluster.
    
    metric: str, default="euclidean"
        The distance metric to use. Passes any option 
        accepted by `scipy.spatial.distance.cdist`.
        
    Returns
    -------
    Hartigen : array-like, shape (n_cluster,)
        The resulting Hartigen index. The optimum number 
        of clusters is the smallest k which produces 
        H(k) ≤ η (typically η = 10). Nevertheless, when
        H(k) > 10, the minimum of the index can be taken 
        as indicating the correct number of clusters in 
        the data.
    
    k_clusters : array-like, shape (n_cluster,)
        List of number of unique labels wrt. kth cluster.
        
    '''
    inputs = np.vstack(dispersion(X, labels, metric)).T
    Hartigen = (-np.diff(inputs[:,0])/inputs[1:,0])
    Hartigen = Hartigen * (X.shape[0]-inputs[:-1,2]-1)
    k_clusters = inputs[:-1,2].astype(int).copy()
    return Hartigen, k_clusters

def get_ax(ax, figsize):

    '''Private: create axis if ax is None'''
    if ax is None: return plt.subplots(figsize=figsize)[1]
    else: return ax
    
def set_annotation(ax, x, y, anno_format, color="blue"):
    
    '''Private: Annotation of y'''
    # Keyword argument.
    kwargs = dict(xytext=(0,6), textcoords='offset points',
                  va='bottom', ha='center', 
                  fontsize=11, color=color, fontweight=500,
                  bbox=dict(facecolor='w', pad=0.5, ec='none'))
    for xy in zip(x,y): 
        ax.annotate(anno_format(xy[1]), xy,**kwargs)

def set_ylim(ax):
    
    '''Private: Set ax.set_ylim()'''
    step = np.diff([n._y for n in ax.get_yticklabels()])[0]
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, max(y_max + step, y_max/0.85)) 

def plot_elbow(score, ax=None, plot_kwds=None, 
               anno_format=None, tight_layout=True):
    
    '''
    Plot Elbow-index.

    Parameters
    ----------
    score : tuple of arrays
        Output from elbow_index function.
    
    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, ax is created 
        with default figsize.
        
    plot_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.plot".

    anno_format : string formatter, default=None
        String formatters (function) for ax.annotate values. 
        If None, it defaults to "{:,.2f}".format.
    
    tight_layout : bool, default=True
        If True, it adjusts the padding between and around 
        subplots i.e. plt.tight_layout().
    
    Returns
    -------
    ax : Matplotlib axis object
        
    '''
    # Create matplotlib.axes if ax is None.
    width = np.fmax(len(score[1])*0.5,6)
    ax = get_ax(ax, (width, 4.3))

    x = np.arange(len(score[1]))
    kwds = dict(lw=1.5, marker='s', ms=8, 
                fillstyle='none', c="blue")
    ax.plot(x, score[0], **(kwds if plot_kwds is 
                            None else {**kwds,**plot_kwds}))
    
    ax.set_xlabel('Number of clusters k', fontsize=11)
    ax.set_ylabel('Gamma, $\gamma(k)$', fontsize=11)
    ax.set_title('Elbow Method\nChoose $k$ '
                 'that maximizes $\gamma(k)$', fontsize=12)
    
    # ax.annotate number format
    set_annotation(ax, x, score[0], 
                   ("{:,.2f}".format if anno_format 
                    is None else anno_format))
    set_ylim(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(score[1])
    if tight_layout: plt.tight_layout()
    return ax
          
def plot_wb(score, ax=None, plot_kwds=None, 
            anno_format=None, tight_layout=True):
    
    '''
    Plot WB-index.
    
    Parameters
    ----------
    score : tuple of arrays
        Output from wb_index function.
    
    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, ax is created 
        with default figsize.

    plot_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.plot".

    anno_format : string formatter, default=None
        String formatters (function) for ax.annotate values. 
        If None, it defaults to "{:,.2f}".format.
    
    tight_layout : bool, default=True
        If True, it adjusts the padding between and around 
        subplots i.e. plt.tight_layout().
    
    Returns
    -------
    ax : Matplotlib axis object
    
    '''
    # Create matplotlib.axes if ax is None.
    width = np.fmax(len(score[1])*0.5,6)
    ax = get_ax(ax, (width, 4.3))
    
    x = np.arange(len(score[1]))
    kwds = dict(lw=1.5, marker='s', ms=8, 
                fillstyle='none', c="blue")
    ax.plot(x, score[0], **(kwds if plot_kwds is 
                            None else {**kwds,**plot_kwds}))
    
    ax.set_xlabel('Number of clusters k', fontsize=11)
    ax.set_ylabel('WB index, $wb(k)$', fontsize=11)
    ax.set_title('WB index\nChoose $k$ '
                 'that minimizes $wb(k)$', fontsize=12)
    
    # ax.annotate number format
    set_annotation(ax, x, score[0], 
                   ("{:,.2f}".format if anno_format 
                    is None else anno_format))
    set_ylim(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(score[1].astype(int))
    if tight_layout: plt.tight_layout()
    return ax
        
def plot_ch(score, ax=None, plot_kwds=None, 
            anno_format=None, tight_layout=True):
    
    '''
    Plot Calinski-Harabasz index.
    
    Parameters
    ----------
    score : tuple of arrays
        Output from calinski_harabasz function.
    
    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, ax is created 
        with default figsize.

    plot_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.plot".

    anno_format : string formatter, default=None
        String formatters (function) for ax.annotate values. 
        If None, it defaults to "{:,.2f}".format.
    
    tight_layout : bool, default=True
        If True, it adjusts the padding between and around 
        subplots i.e. plt.tight_layout().
    
    Returns
    -------
    ax : Matplotlib axis object
    
    '''
    # Create matplotlib.axes if ax is None.
    width = np.fmax(len(score[1])*0.5,6)
    ax = get_ax(ax, (width, 4.3))
    
    x = np.arange(len(score[1]))
    kwds = dict(lw=1.5, marker='s', ms=8, 
                fillstyle='none', c="blue")
    ax.plot(x, score[0], **(kwds if plot_kwds is 
                            None else {**kwds,**plot_kwds}))
    
    ax.set_xlabel('Number of clusters k', fontsize=11)
    ax.set_ylabel('Calinski-Harabasz index, $ch(k)$', fontsize=11)
    ax.set_title('Calinski Harabasz index\nChoose $k$ '
                 'that maximizes $ch(k)$', fontsize=12)
    
    # ax.annotate number format
    set_annotation(ax, x, score[0], 
                   ("{:,.2f}".format if anno_format 
                    is None else anno_format))
    set_ylim(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(np.array(score[1]).astype(int))
    if tight_layout: plt.tight_layout()
    return ax
        
def plot_hartigan(score, ax=None, plot_kwds=None, 
                  anno_format=None, tight_layout=True):
    
    '''
    Plot Hartigan-index.
    
    Parameters
    ----------
    score : tuple of arrays
        Output from hartigen_index function.
    
    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, ax is created 
        with default figsize.

    plot_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.plot".

    anno_format : string formatter, default=None
        String formatters (function) for ax.annotate values. 
        If None, it defaults to "{:,.2f}".format.
    
    tight_layout : bool, default=True
        If True, it adjusts the padding between and around 
        subplots i.e. plt.tight_layout().
    
    Returns
    -------
    ax : Matplotlib axis object
    
    '''
    # Create matplotlib.axes if ax is None.
    width = np.fmax(len(score[1])*0.5,6)
    ax = get_ax(ax, (width, 4.3))
    
    x = np.arange(len(score[1]))
    kwds = dict(lw=1.5, marker='s', ms=8, 
                fillstyle='none', c="blue")
    ax.plot(x, score[0], **(kwds if plot_kwds is 
                            None else {**kwds,**plot_kwds}))
    
    ax.set_xlabel('Number of clusters k', fontsize=11)
    ax.set_ylabel('Hartigen index, $H(k)$', fontsize=11)
    ax.set_facecolor('white')
    ax.set_title('Hartigen index\nChoose the smallest '
                 '$k$ such that $H(k) \leq \eta$', fontsize=12)
    
    # ax.annotate number format
    set_annotation(ax, x, score[0], 
                   ("{:,.2f}".format if anno_format 
                    is None else anno_format))
    set_ylim(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(np.array(score[1]).astype(int))
    if tight_layout: plt.tight_layout()
    return ax
      
def plot_silhouette(score, ax=None, plot_kwds=None, 
                    anno_format=None, tight_layout=True):
    
    '''
    Plot Silhouette-score.
    
    Parameters
    ----------
    score : tuple of arrays
        Output from silhouette function.
    
    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, ax is created 
        with default figsize.

    plot_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.plot".

    anno_format : string formatter, default=None
        String formatters (function) for ax.annotate values. 
        If None, it defaults to "{:,.2f}".format.
    
    tight_layout : bool, default=True
        If True, it adjusts the padding between and around 
        subplots i.e. plt.tight_layout().
    
    Returns
    -------
    ax : Matplotlib axis object
    
    '''
    # Create matplotlib.axes if ax is None.
    width = np.fmax(len(score[1])*0.5,6)
    ax = get_ax(ax, (width, 4.3))
    
    x = np.arange(len(score[1]))
    kwds = dict(lw=1.5, marker='s', ms=8, 
                fillstyle='none', c="blue")
    ax.plot(x, score[0], **(kwds if plot_kwds is 
                            None else {**kwds,**plot_kwds}))
    
    ax.set_xlabel('Number of clusters k', fontsize=11)
    ax.set_ylabel('Average Silhouette width, $sil(k)$', fontsize=11)
    ax.set_title('Silhouette\nChoose $k$ that '
                 'maximizes $sil(k)$', fontsize=12)
    
    # Criteria line.
    kwargs = dict(xytext=(0,-1), textcoords='offset points',
                  va='top', ha='right', fontsize=10, 
                  color='#485460')
    criteria = [('Artificial $(\geq0.25)$',0.25),
                ('Reasonable $(\geq0.50)$',0.50),
                ('Strong $(\geq0.70)$',0.70)]
    for c in criteria: ax.annotate(c[0],(x[-1],c[1]),**kwargs)
    kwargs = dict(color='#485460', lw=1, ls='--')
    for n in [0.25,0.5,0.7]: ax.axhline(n, **kwargs)

        
    # ax.annotate number format
    set_annotation(ax, x, score[0], 
                   ("{:,.2f}".format if anno_format 
                    is None else anno_format))
    set_ylim(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(np.array(score[1]).astype(int))
    ax.set_xlim(-0.5,len(x)-0.5)
    if tight_layout: plt.tight_layout()
    return ax  
           
def plot_gap(score, ch_type=1, ax=None, 
             plot_kwds=None, errorbar_kwds=None,
             anno_format=None, tight_layout=True):
    
    '''
    Plot Gap-Statistics.
    
    Parameters
    ----------
    score : tuple of arrays
        Output from hartigen_index function.
    
    ch_type : int, default=1
        1 : It plots Log(W(k)) and E{Log(W(kb))}
            (reference distribution).
        2 : It plots Gap-Statistics of all `k` group
            i.e. Gap(k) = E{Log(W(kb))} - Log(W(k)).
        3 : It plots Delta(k), which is defined as
            Delta(k) = Gap(k) - Gap(k+1) + s(k+1).
    
    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, ax is created 
        with default figsize.

    plot_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.plot".
    
    errorbar_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.errorbar".
    
    anno_format : string formatter, default=None
        String formatters (function) for ax.annotate values. 
        If None, it defaults to "{:,.2f}".format.
    
    tight_layout : bool, default=True
        If True, it adjusts the padding between and around 
        subplots i.e. plt.tight_layout().
    
    Returns
    -------
    ax : Matplotlib axis object
    
    '''
    # Create matplotlib.axes if ax is None.
    width = np.fmax(len(score[1])*0.5,6)
    ax = get_ax(ax, (width, 4.3))
    
    x = np.arange(len(score[3]))
    
    if ch_type==1:
        
        # Compare Log(W(k)) against E{Log(W(kb))} reference distribution.
        kwds = dict(lw=1.5, marker='s', ms=8, fillstyle='none')
        plot_dict = [{'color':'b','label':r'$\log{(W_{k})}$'},
                     {'color':'k','label':r'$E\{\log{(W_{kb})}\}$'}]
        for n in [0,1]:
            kwds.update(plot_dict[n])
            ax.plot(x, score[n], **(kwds if plot_kwds is 
                                    None else {**kwds,**plot_kwds}))
        
            # ax.annotate number format
            set_annotation(ax, x, score[n], 
                           ("{:,.2f}".format if anno_format 
                            is None else anno_format), 
                           plot_dict[n]["color"])

        ax.set_ylabel(r'Intra-cluster distance = $\log{(W_{k})}$', fontsize=11)
        ax.set_title('Gap-Statistics\nComparison of $W_{k}$ with null '
                     'reference distribution $W_{kb}$', fontsize=12)
        ax.legend(loc='best', framealpha=0)
    
    elif ch_type==2:
        
        #  Gap(k) = E{Log(W(kb))} - Log(W(k))
        kwds = dict(elinewidth=1, lw=1.5, marker='s', ms=8, 
                    fillstyle='none', color="blue")
        ax.errorbar(x, score[1]-score[0], yerr=score[2]*2, 
                    **(kwds if errorbar_kwds is None else {**kwds,**errorbar_kwds}))
        
        ax.set_ylabel(r'$Gap(k) = E\{\log{(W_{kb})}\}-\log{(W_{k})}$', fontsize=11)
        ax.set_title('Gap-Statistics\nChoose $k$ such that '
                     '$Gap(k)$ is maximized.', fontsize=12)
        
        set_annotation(ax, x, score[1]-score[0], 
                       ("{:,.2f}".format if anno_format 
                        is None else anno_format))
        
    elif ch_type==3:
        
        # Delta(k) = Gap(k) - Gap(k+1) + s_{k+1}
        gaps = score[1]-score[0]
        gaps = np.hstack((gaps[:-1]-gaps[1:]+score[2][1:],[0]))
        kwds = dict(lw=1.5, marker='s', ms=8, fillstyle='none', color="blue")
        ax.plot(x, gaps, **(kwds if plot_kwds is 
                            None else {**kwds,**plot_kwds}))
        
        ax.set_ylabel(r'$\Delta = Gap(k) - Gap(k+1) + s_{k+1}$', fontsize=11)
        ax.set_title('Gap-Statistics\nChoose the smallest $k$ such that '
                     '$\Delta \geq0$', fontsize=12)
        ax.axhline(0, lw=2, c='grey', ls='-')
        set_annotation(ax, x, gaps, ("{:,.2f}".format if anno_format 
                                     is None else anno_format))
     
    set_ylim(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(np.array(score[3]).astype(int))
    ax.set_xlabel('Number of clusters k', fontsize=11)
    if tight_layout: plt.tight_layout()
    return ax