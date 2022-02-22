'''
Available methods are the followings:
[1] dispersion
[2] calinski_harabasz
[3] elbow_index
[4] silhouette
[5] wb_index
[6] gap_statistics
[7] hartigan_index
[8] dudahart_index
[9] eval_cluster

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 21-02-2022

'''
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import collections
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.transforms as transforms
from matplotlib.ticker import(FixedLocator, 
                              FixedFormatter, 
                              StrMethodFormatter,
                              FuncFormatter)
from matplotlib.patches import Patch, FancyArrowPatch
from matplotlib.lines import Line2D
from functools import partial

plt.rcParams.update({'font.family':'sans-serif'})
plt.rcParams.update({'font.sans-serif':'Hiragino Sans GB'})
plt.rc('axes', unicode_minus=False)

__all__ = ['dispersion', 
           'calinski_harabasz', 
           'elbow_index', 
           'silhouette', 
           'wb_index', 
           'gap_statistics', 
           'hartigan_index', 
           'dudahart_index',
           'eval_cluster',
           'plot_ch_base',
           'plot_elbow_base',
           'plot_wb_base',
           'plot_silhouette_base',
           'plot_gap1_base',
           'plot_gap2_base',
           'plot_hartigan_base',
           'plot_dh_base']

def dispersion(X, labels, metric='euclidean'):
    
    '''
    Determine the dispersion within cluster (SSW) and between clusters 
    (SSB), which are defined as follows:
    
                        SSW = ∑ (‖ x, C ‖^2 | x ∈ C)

                        SSB = ∑ (‖ C, μ ‖^2 | C ∈ K)

    where ‖ x, C ‖ is a distance from instance to its cluster. Such 
    distance can be any metric. K is a set of clusters {C1, C2, ..., 
    Cn} from X and μ is a centroid of X.
    
    Parameters
    ----------
    X : 2d-array, shape of (n_sample, n_feature)
        Training instances. Each row corresponds to a single data 
        point.
        
    labels : 2d-array, shape of (n_sample, n_cluster)
        Predicted labels of instances for respective n_cluster or kth 
        cluster.
    
    metric: str, default="euclidean"
        The distance metric to use. Passes any option accepted by 
        "scipy.spatial.distance.cdist".
    
    Returns
    -------
    intra_disp : array-like, shape (n_cluster,)
        List of within-cluster dispersions wrt. kth cluster.
    
    extra_disp : array-like, shape (n_cluster,)
        List of between-clusters dispersions wrt. kth cluster. 
    
    k_clusters : array-like, shape (n_cluster,)
        Number of unique labels wrt. kth cluster.
        
    '''
    # Initialize loop parameters.
    intra_disp, extra_disp, k_clusters = [], [], []
    mean = np.mean(X, axis=0) # Center of X.
    X = np.array(X).copy()
    
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

class eval_cluster():
    
    '''
    `eval_cluster` provides a quick access to all evaluation methods 
    for clustering under "cluster_validity.py". Moreover, it also allows 
    adjustment or modification to be made to any particular plot.
    
    Parameters
    ----------
    metric: str, default="euclidean"
        The distance metric to use. Passes any option accepted by 
        "scipy.spatial.distance.cdist".
    
    p_sample : float, default=0.5
        Percent of samples ("silhouette" and "gap_statistics").
    
    n_bootstraps : int, default=5
        Number of bootstrappings ("gap_statistics").
        
    random_state : int, defualt=0
        Controls both the randomness of the bootstrapping of the samples 
        used in "gap_statistics" and the sampling based on `p_sample`.
        
    weights : array-like, of shape (n_methods,), default=None
        Weights for respective methods. The order must correspond to 
        that in `methods` (attribute), which is a list of methods.
            
    Attributes
    ----------
    results : collections.namedtuple
        A dict with keys as validation method and values as result 
        from corresponding function.
        - 'ch': Calinski and Harabasz index ("calinski_harabasz")
        - 'e' : Elbow ("elbow_index")
        - 's' : Silhouette coefficient ("silhouette")
        - 'w' : WB index ("wb_index")
        - 'g' : Gap statistic ("gap_statistics")
        - 'h' : Hartigan index ("hartigan_index")
        - 'dh': Duda and Hart index ("dudahart_index")
            
    '''
    def __init__(self, metric='euclidean', p_sample=0.5, n_bootstraps=5, 
                 random_state=0, weights=None):
        
        self.metric = metric
        self.p_sample = p_sample
        self.n_bootstraps = n_bootstraps
        self.random_state = random_state
        
        # Validation functions
        kwds = {"n" : dict(metric=metric)}
        kwds["s"] = {**kwds["n"], **dict(random_state=random_state, 
                                         p_sample=p_sample)}
        kwds["g"] = {**kwds["s"], **dict(n_bootstraps=n_bootstraps)}
        self.funcs = {"ch": partial(calinski_harabasz, **kwds["n"]), 
                      "e" : partial(elbow_index, **kwds["n"]), 
                      "s" : partial(silhouette, **kwds["s"]), 
                      "w" : partial(wb_index, **kwds["n"]), 
                      "g" : partial(gap_statistics, **kwds["g"]), 
                      "h" : partial(hartigan_index, **kwds["n"]),
                      "dh": partial(dudahart_index, **kwds["n"])}
        
        # Plot functions.
        self.fnc_plots = {"ch": plot_ch_base, 
                          "e" : plot_elbow_base, 
                          "s" : plot_silhouette_base, 
                          "w" : plot_wb_base, 
                          "g1": plot_gap1_base, 
                          "g2": plot_gap2_base, 
                          "h" : plot_hartigan_base, 
                          'dh': plot_dh_base}
        
        self.methods = list(self.fnc_plots.keys())
        n_methods = len(self.methods)
        if weights is not None: 
            self.weights  = np.array(weights).ravel()
            self.weights /= sum(self.weights)
        else: self.weights = np.ones(n_methods)/n_methods
        
    def fit(self, X, labels):
        
        '''
        Fit model.
        
        Parameters
        ----------
        X : 2d-array, shape of (n_samples, n_features)
            Training instances. Each row corresponds to a single data 
            point.

        labels : 2d-array, shape of (n_samples, n_clusters)
            Predicted labels of instances for respective n_clusters or kth 
            cluster.
            
        Attributes
        ----------
        results : collections.namedtuple
            A dict with keys as validation method and values as result 
            from corresponding function.
            - 'ch': Calinski and Harabasz index ("calinski_harabasz")
            - 'e' : Elbow ("elbow_index")
            - 's' : Silhouette coefficient ("silhouette")
            - 'w' : WB index ("wb_index")
            - 'g' : Gap statistic ("gap_statistics")
            - 'h' : Hartigan index ("hartigan_index")
            - 'dh': Duda and Hart index ("dudahart_index")
            
        w_scores : 1d-array, shape of (n_clusters,)
            An array of weighted scores.

        '''
        Results = collections.namedtuple("Results", self.funcs.keys())
        self.results = Results(**dict((key,fnc(X,labels)) for 
                                      key,fnc in self.funcs.items()))
        
        # Cluster indices.
        self.index = np.r_[[len(np.unique(labels[:,n])) 
                            for n in range(labels.shape[1])]]
        
        # Calculate weighted scores.
        self.scores = self.__cal_score__()
        w = self.weights.reshape(-1,1)
        self.w_scores = np.nan_to_num(self.scores).dot(w).ravel()

        return self
    
    def __cal_score__(self):
        
        '''
        Calculate weighted scores.

        Returns
        -------
        scores : 2d-array, shape of (n_clusters, n_methods)
            The order of the results corresponds to `methods`.
        
        '''
        # Score array.
        scores = np.full((len(self.index), 
                          len(self.methods)), np.nan)
        
        for (i,method) in enumerate(self.methods):

            if method in ['g1','g2']:
                r = getattr(self.results, "g")
            else: r = getattr(self.results, method)
            
            # Get scores.
            if method == 'g1': score = r[3]
            elif method == 'g2': score = r[4]
            else: score = r[0]
            
            # Cap scores
            if method == 'h' :
                score = np.where(score>np.mean(score), 0, score)
            elif method == 'dh':
                score = np.where(score<r[2], 0, score)
        
            # Normalize scores.
            amin, amax = np.percentile(score, q=[0,100])
            score = (score-amin)/(amax-amin)
            if method=='w': score = 1 - score
                
            # Store scores in array.
            if method in ['g1','g2']: index = np.array(r[5])-1
            else: index = np.array(r[1])-1  
            scores[index.astype(int), i] = score
                
        return scores
    
    def plot(self, method="ch", ax=None, color="#FC427B", plot_kwds=None, 
             show_anno=True, anno_format=None, tight_layout=True):
        
        '''
        Plot result of specified `method`.
   
        Parameters
        ----------
        method : str, default='ch'
            Specify the method of validation:
            - 'ch': Calinski and Harabasz index
            - 'e' : Elbow (gamma)
            - 's' : Silhouette coefficient
            - 'w' : WB index
            - 'g1': Gap statistic, Gap(k)
            - 'g2': Gap statistic, Delta(k)
            - 'h' : Hartigan index
            - 'dh': Duda and Hart index
            - 'ws': Weighted scores

        ax : Matplotlib axis object, default=None
            Predefined Matplotlib axis. If None, ax is created with 
            default figsize.

        color : Color-hex, default="#FC427B"
            Color to be passed to "ax.plot". This overrides "plot_kwds".

        plot_kwds : keywords, default=None
            Keyword arguments to be passed to "ax.plot".

        show_anno : bool, default=True
            If True, it annotates the point xy with y.

        anno_format : string formatter, default=None
            String formatters (function) for ax.annotate values. If None, 
            it defaults to "{:,.2f}".format.

        tight_layout : bool, default=True
            If True, it adjusts the padding between and around subplots 
            i.e. plt.tight_layout().
            
        Returns
        -------
        ax : Matplotlib axis object
    
        '''
        # Check whether `method` is a valid key.
        all_methods = self.methods + ['ws']
        if method not in all_methods:
            raise ValueError(f"Invalid `method`. It only accepts" 
                             f" method in {all_methods}. Got " 
                             f"'{method}' instead.")
        elif method != 'ws':
            
            # Get scores from `self.results`.
            name = "g" if method in ["g1", "g2"] else method
            score = getattr(self.results, name)
       
        # Keyword arguments for plot functions.
        kwargs = dict(ax=ax, color=color, 
                      plot_kwds=plot_kwds, 
                      show_anno=show_anno, 
                      anno_format=anno_format, 
                      tight_layout=tight_layout)
        if method == 'ws': ax = self.plot_w_score(**kwargs)
        else: ax = self.fnc_plots[method](score, **kwargs)

        return ax
    
    def plot_w_score(self, ax=None, color="#FC427B", plot_kwds=None, 
                     show_anno=True, anno_format=None, tight_layout=True):
        
        '''Private function: plot `w_score`'''
        ax = get_ax(ax, self.w_scores)
        x  = np.arange(len(self.index))
        
        kwds = dict(linewidth=5, solid_capstyle='round', 
                    solid_joinstyle="round")
        if plot_kwds is not None: kwds.update(plot_kwds)
        kwds.update({"color":color})
        ax.plot(x, self.w_scores, **kwds)

        if show_anno: set_annotate(ax, x, self.w_scores, color, anno_format)
        set_yaxis(ax, "Score", r"$WS(k)$")
        xlabel = r'Weighted Score : Choose $k$ that maximizes $WS(k)$'
        set_xaxis(ax, x, self.index, xlabel, "$k$")
        set_axvline(ax, np.argmax(self.w_scores), "$k$*", color)
        if tight_layout: plt.tight_layout()

def calinski_harabasz(X, labels, metric='euclidean'):
    
    '''
    Compute the Calinski and Harabasz index (CH), which is defined as 
    follows:
    
                    CH = [SSB.(N - K)]/[SSW.(K - 1)]
    
    where `SSW` is the within-group dispersion matrix for data 
    clustered into `K` clusters, while `SSB` is a dispersion between 
    clusters. `N` is a number of samples (`X`), and `K` is a number of 
    clusters.
    
    References
    ----------
    .. [1] https://www.researchgate.net/publication/257138057_An_exami
           nation_of_indices_for_determining_the_number_of_clusters_Nb
           Clust_Package
    .. [2] https://scikit-learn.org/stable/modules/generated/sklearn.
           metrics.calinski_harabasz_score.html
        
    Parameters
    ----------
    X : 2d-array, shape of (n_samples, n_features)
        Training instances. Each row corresponds to a single data 
        point.
        
    labels : 2d-array, shape of (n_samples, n_clusters)
        Predicted labels of instances for respective n_clusters or kth 
        cluster.
    
    metric: str, default="euclidean"
        The distance metric to use. Passes any option accepted by 
        "scipy.spatial.distance.cdist".
        
    Returns
    -------
    score : array-like, shape (n_clusters,)
        The resulting Calinski-Harabasz index. The maximum value of 
        the index is taken as indicating the correct number of 
        clusters in the data.
    
    k_clusters : array-like, shape (n_clusters,)
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
    This method looks at change of within-group dispersion (`SSW`). 
    One should choose a number of clusters so that adding another 
    cluster doesn't reduce significant amount of `SSW`. 
    
    More precisely, if one plots `SSW` by the clusters against the 
    number of clusters (`Elbow plot`), one should choose the point 
    before marginal gain (ΔSSW) becomes minute or insignificant. Thus, 
    calculating 2nd order of relative change in distance function is 
    introduced to help detect such point, which can be expressed as 
    follows:
    
                        W = ∑ (‖ x, C ‖^2 | x ∈ C)

                      γ(k) = -(δ(k) - δ(k-1))/δ(k-1)

                      δ(k) = (W(k) - W(k-1))/W(k-1)
    
    where ‖ x, C ‖ is a distance from instance to its cluster. Such 
    distance can be any metric. k is a  set of clusters {C1, C2, ..., 
    Cn} from X and Gamma is the  rate of change. 
  
    Parameters
    ----------
    X : 2d-array, shape of (n_samples, n_features)
        Training instances. Each row corresponds to a single data 
        point.
        
    labels : 2d-array, shape of (n_samples, n_clusters)
        Predicted labels of instances for respective n_clusters or kth 
        cluster. The number of n_clusters must be more than 3.
    
    metric: str, default="euclidean"
        The distance metric to use. Passes any option  accepted by 
        "scipy.spatial.distance.cdist".

    Returns
    -------
    Gamma : array-like, shape (n_clusters-2,)
        The resulting rate of change in distances. The maximum value 
        of the index is taken as indicating the correct number of 
        clusters in the data.
    
    k_clusters : array-like, shape (n_clusters-2,)
        List of number  of unique labels wrt. kth cluster.
        
    '''
    # 2nd derivative of the distances.
    inputs = np.vstack(dispersion(X, labels, metric)).T
    Delta = np.diff(inputs[:,0])/inputs[:-1,0]
    Gamma = np.diff(Delta)/Delta[:-1]
    k_clusters = inputs[1:-1,2].astype(int).tolist()
    return -Gamma, k_clusters

def silhouette(X, labels, metric='euclidean', p_sample=0.5, 
               random_state=None):
    
    '''
    Silhouette coefficient is calculated using the mean intra-cluster 
    distance (a) and the mean nearest-cluster distance (b) for each 
    sample. The formula (for a sample) is expressed as follows: 
    
                    Silhouette = (b - a) / max{a, b}
  
    where `a` is a mean intra-cluster, and `b` is a mean nearest-
    cluster distance.
    
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Silhouette_(clustering)
    .. [2] https://scikit-learn.org/stable/auto_examples/cluster/
           plot_kmeans_silhouette_analysis.html
        
    Parameters
    ----------
    X : 2d-array, shape of (n_samples, n_features)
        Training instances. Each row corresponds to a single data 
        point.
        
    labels : 2d-array, shape of (n_samples, n_clusters)
        Predicted labels of instances for respective n_clusters or kth 
        cluster.
    
    metric: str, default="euclidean"
        The distance metric to use. Passes any option  accepted by 
        "scipy.spatial.distance.cdist".
    
    p_sample : float, default=0.5
        Percent of samples.
    
    random_state : int, defualt=None
        Random state for the random number generator.
    
    Returns
    -------
    score : array-like, shape (n_clusters,)
        The resulting silhouette coefficients. The maximum value of 
        the index is taken as indicating the correct number of 
        clusters in the data.

    k_clusters : array-like, shape (n_clusters,)
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
    Compute the WB-index (WB), which is defined as follows:

                            WB = K.SSW/SSB
    
    where SSW, SSB, and K are a dispersion within cluster, a 
    dispersion between clusters, and number of clusters, respectively.
  
    Parameters
    ----------
    X : 2d-array, shape of (n_samples, n_features)
        Training instances. Each row corresponds to a single data 
        point.
        
    labels : 2d-array, shape of (n_samples, n_clusters)
        Predicted labels of instances for respective n_clusters or kth 
        cluster.
    
    metric: str, default="euclidean"
        The distance metric to use. Passes any option  accepted by 
        "scipy.spatial.distance.cdist".
    
    Returns
    -------
    WB : array-like, shape (n_clusters,)
        The resulting WB indices in logarithmic form, LOG(WB). The 
        minimum value of the index is taken as indicating the correct 
        number of clusters in the data.
    
    k_clusters : array-like, shape (n_clusters,)
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
                   n_bootstraps=5, random_state=None):
    
    '''
    The idea of the Gap statistic is to compare the SSW (dispersion) 
    to its expectation under an appropriate null reference 
    distribution i.e. a random uniform distribution. It can be 
    mathematically expressed as:
    
                    Gap(k) = Log(E[SSW(k,n)]) - Log(SSW)
            
    where k and n respresent number of clusters and number of 
    bootstrappings, respectively. This function uses `sklearn.cluster.
    KMeans` to determine cluster centroids from a set of randomly 
    distributed datasets.
    
    The optimum number of clusters is the smallest value k such that 
    Gap(k) ≥ Gap(k+1) − s(k+1), where s(k) is a factor that takes into 
    account the standard deviation of the Monte-Carlo replicates SSW.
    
    References
    ----------
    .. [1] https://statweb.stanford.edu/~gwalther/gap
    .. [2] https://datasciencelab.wordpress.com/tag/gap-statistic/
    
    Parameters
    ----------
    X : 2d-array, shape of (n_samples, n_features)
        Training instances. Each row corresponds to a single data 
        point.
        
    labels : 2d-array, shape of (n_samples, n_clusters)
        Predicted labels of instances for respective n_clusters or kth 
        cluster.
    
    metric: str, default="euclidean"
        The distance metric to use. Passes any option  accepted by 
        "scipy.spatial.distance.cdist".
    
    p_sample : float, default=0.5
        Percent of samples.
    
    n_bootstraps : int, default=5
        Number of bootstrappings for Gap-statistics.
    
    random_state : int, defualt=None
        Random state for the random number generator.
    
    Returns
    -------
    Wks : array-like, shape (n_clusters,)
        The resulting LOG(Wk) from X. 
    
    Wkbs : array-like, shape (n_clusters,)
        The resulting E[LOG(Wk)] from a set of random uniform 
        distribution of X.

    sk : array-like, shape (n_clusters,)
        Standard deviation from a set of random uniform distribution 
        of X.
        
    Gaps : array-like, shape (n_clusters,)
        Gap(k) = E{Log(W(kb))} - Log(W(k))
        
    Deltas : array-like, shape (n_clusters,)
        Delta(k) = Gap(k) - Gap(k+1) + sk(k+1)
    
    k_clusters : array-like, shape (n_clusters,)
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
    rand = np.random.RandomState(random_state)
    n_sample = int(p_sample*X.shape[0])
    Wks = np.zeros(labels.shape[1])
    Wkbs = np.zeros(labels.shape[1])
    sk = np.zeros(labels.shape[1])
    
    # Create 'n' bootstapped dataset(s) with random uniform 
    # distribution for all features in 'X'.
    kwargs = [{'low':a, 'high':b, 'size':n_sample*n_bootstraps} 
              for (a,b) in zip(np.min(X,axis=0), np.max(X,axis=0))]
    bsX = [rand.uniform(**k).reshape(-1,1) for k in kwargs]
    bsX = np.split(np.hstack(bsX), n_bootstraps, axis=0)
     
    for k in range(labels.shape[1]):
        
        # Compute 'Wk' for respective kth clusters.
        Wks[k] = np.log(intra_dist(X, labels[:,k], metric))
        
        # Loop parameters.
        BWkbs = np.zeros(n_bootstraps)
        kwargs = {'n_clusters' : len(np.unique(labels[:,k])), 
                  'random_state' : random_state}
        
        # Main loop for bootstrapped datasets.
        for n in range(n_bootstraps):
            
            # Compute within-cluster sum of squares for all  
            # clusters given 'kth' clusters.
            args = (bsX[n], KMeans(**kwargs).fit_predict(bsX[n]))
            BWkbs[n] = np.log(intra_dist(*args, metric=metric))
    
        # Wk is the pooled within-cluster sum of squares around  
        # the cluster means from differenct bootstrapped Xs.
        Wkbs[k] = sum(BWkbs)/n_bootstraps
        
        # Compute standard deviation.
        sk[k] = np.sqrt(sum((BWkbs-Wkbs[k])**2)/n_bootstraps)
        sk[k] = sk[k]*np.sqrt(1+1/n_bootstraps)
    
    # Number of clusters as per kth cluster.
    k_clusters = np.array([np.unique(labels[:,n]).shape[0] 
                           for n in range(labels.shape[1])])
    
    # Gap(k) = E{Log(W(kb))} - Log(W(k))
    Gaps = Wkbs - Wks
    
    # Delta(k) = Gap(k) - Gap(k+1) + s{k+1}
    Deltas = np.r_[(Gaps[:-1] - Gaps[1:] + sk[1:]), 0]
    
    return Wks, Wkbs, sk, Gaps, Deltas, k_clusters

def hartigan_index(X, labels, metric='euclidean'):
    
    '''
    Compute the Hartigan index (1975), which is defined as follows:
    
                H(k) = (SSW(k)/SSW(k+1) - 1).(N - k - 1)

    where `SSW` is the within-group dispersion matrix for data 
    clustered into `k` clusters. `N` is a number of samples, and `k` 
    is a number of clusters.

    Parameters
    ----------
    X : 2d-array, shape of (n_samples, n_features)
        Training instances. Each row corresponds to a single data 
        point.
      
    labels : 2d-array, shape of (n_samples, n_clusters)
        Predicted labels of instances for respective n_clusters or kth 
        cluster.
    
    metric: str, default="euclidean"
        The distance metric to use. Passes any option accepted by 
        "scipy.spatial.distance.cdist".
        
    Returns
    -------
    Hartigan : array-like, shape (n_clusters,)
        The resulting Hartigan index. The optimum number of clusters 
        is the smallest k which produces H(k) ≤ η (typically η = 10). 
        Nevertheless, when H(k) > 10, the minimum of the index can be 
        taken as indicating the correct number of clusters in the 
        data.
    
    k_clusters : array-like, shape (n_clusters-1,)
        List of number of unique labels wrt. kth cluster.
        
    '''
    inputs = np.vstack(dispersion(X, labels, metric)).T
    Hartigan = (-np.diff(inputs[:,0])/inputs[1:,0])
    Hartigan = Hartigan * (X.shape[0]-inputs[:-1,2]-1)
    k_clusters = inputs[:-1,2].astype(int).copy()
    return Hartigan, k_clusters

def dudahart_index(X, labels, metric='euclidean'):
    
    '''
    Duda and Hart (1973) suggested the ratio of the two within sum of 
    squares to decide whether a cluster can be divided into two 
    clusters is as follows
    
                 DH(k) = SSW(k+1)/SSW(k) where k>0

    where `SSW` is the within-group dispersion matrix for data 
    clustered into `k` clusters, and `k` is a number of clusters. The 
    following criterion was proposed to sub-divide whenever the 
    following holds
    
              DH(k) < 1 - 2/πp - z sqrt(2(1 - 8/(π^2p))/(np))

    Parameters
    ----------
    X : 2d-array, shape of (n_samples, n_features)
        Training instances. Each row corresponds to a single data 
        point.
      
    labels : 2d-array, shape of (n_samples, n_clusters)
        Predicted labels of instances for respective n_clusters or kth 
        cluster.
    
    metric: str, default="euclidean"
        The distance metric to use. Passes any option accepted by 
        "scipy.spatial.distance.cdist".
        
    Returns
    -------
    dudahart : array-like, shape (n_clusters-1,)
        The resulting Duda and Hart Index. 
    
    k_clusters : array-like, shape (n_clusters,)
        List of number of unique labels wrt. kth cluster.
        
    threshold : float
        
    '''
    inputs = np.vstack(dispersion(X, labels, metric)).T
    SSW, z, (n,p) = inputs[:,0], 3.2, X.shape
    dudahart = SSW[1:] / SSW[:-1]
    c = z * np.sqrt(2*(1 - 8/(pow(np.pi,2)*p)) / (n*p))
    threshold = 1-2/(np.pi*p) - c
    k_clusters = inputs[:-1,2].astype(int).copy()
    return dudahart, k_clusters, threshold

def set_yaxis(ax, label, text):
    
    '''Private function: ax.yaxis'''
    y_min, y_max = ax.get_ylim()
    diff = 0.1 * (y_max - y_min)
    ax.set_ylim(y_min-diff, y_max+diff)
    ax.set_ylabel(label, fontsize=13)
    
    args = (ax.transAxes, ax.transAxes)
    trans = transforms.blended_transform_factory(*args)
    ax.text(0, 1.01, text, transform=trans, fontsize=13, 
            va='bottom', ha="center")
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(6))
    ax.yaxis.set_tick_params(labelsize=11)
    
def set_xaxis(ax, x, ticklabels, label, text):
    
    '''Private function: ax.xaxis'''
    ax.set_xlabel(label, fontsize=13)
    args = (ax.transAxes, ax.transAxes)
    trans = transforms.blended_transform_factory(*args)
    ax.text(1.01, 0, text, transform=trans, fontsize=13, 
            va='center', ha="left")
    ax.set_xticks(x)
    ax.set_xticklabels(ticklabels)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_facecolor('white')
    ax.patch.set_alpha(0)
    ax.xaxis.set_tick_params(labelsize=11)
    
def set_axvline(ax, x, text, color):
    
    '''Private function: ax.axvline'''
    ax.axvline(x, color="#cccccc", lw=0.8, zorder=-1)
    args = (ax.transData, ax.transAxes)
    trans = transforms.blended_transform_factory(*args)
    ax.text(x, 1.01, text, transform=trans, fontsize=13, 
            va="bottom", ha="center", color=color)
    
def get_ax(ax, x):
    
    '''Private function: plt.subplots'''
    width = np.fmax(len(x)*0.5,6)
    if ax is None: ax = plt.subplots(figsize=(width, 4.3))[1]
    return ax

def set_annotate(ax, x, y, color, num_format=None):
    
    '''Private function: ax.annotate'''
    # Keyword argument.
    if num_format is None: num_format = "{:,.2f}".format 
    default = dict(textcoords='offset points', fontsize=12, color=color,
                   bbox=dict(facecolor="w", pad=0., edgecolor='none'),
                   arrowprops = dict(arrowstyle = "-", color=color))
    vectors, degrees, quad_left, quad_right = vector_coord(ax, x, y)
    
    c, gap = 15, 5
    for n,xy in enumerate(zip(x,y)): 
        
        left, right = quad_left[n], quad_right[n]
        deg, (x0, y0) = degrees[n], vectors[n]
        kwds = dict(xytext=(x0*c,y0*c), ha='center', va='center')
        
        if ((n==0) & (right==0)) | ((n==len(x)-1) & (left==1)):
            kwds = {"xytext": (0,-10),"ha": 'center',"va": 'top'}   
        elif ((n==0) & (right==3)) | ((n==len(x)-1) & (left==2)):
            kwds = {"xytext": (0, 10),"ha": 'center',"va": 'bottom'}      
        elif (left==2) & (right==3):
            kwds = {"xytext": (0, 10),"ha": 'center',"va" : 'bottom'}         
        elif (left==1) & (right==0):
            kwds = {"xytext": (0, -10),"ha": 'center',"va": 'top'}        
        elif (deg>=90-gap) & (deg<=90+gap):
            kwds = {"xytext": (x0*c,y0*c),"ha": 'center',"va": 'bottom'}         
        elif (deg>=270-gap) & (deg<=270+gap):
            kwds = {"xytext": (x0*c,y0*c),"ha": 'center',"va": 'top'}       
        elif (deg>90+gap) & (deg<270-gap):
            kwds = {"xytext": (x0*c,y0*c),"ha": 'right',"va": 'center'}       
        elif (deg<90-gap) | (deg>270+gap):
            kwds = {"xytext": (x0*c,y0*c),"ha": 'left',"va": 'center'} 
        ax.annotate(num_format(xy[1]), xy, **{**default,**kwds})

def extrapolate(x, y, forward=True):
    
    '''Private Function: extrapolation'''
    slope = np.diff(y) / np.diff(x)
    const = y[1] - slope * x[1]
    if forward: expl_x = x[1] + np.diff(x)
    else: expl_x = (x[0] - np.diff(x))
    expl_y = slope * expl_x + const
    return float(expl_x), float(expl_y)

def vector_coord(ax, x, y):
    
    '''Private Function: determine unit vector coordinates'''
    def find_quadrant(x,y):
        if (x>=0) & (y>=0): return 0
        elif (x<0) & (y>=0): return 1
        elif (x<0) & (y<0): return 2
        else: return 3
    
    # Transform x and y to axis coordinate system.
    renderer = plt.gcf().canvas.get_renderer()
    bbox_ax  = ax.get_window_extent(renderer=renderer)
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    norm_x = (np.array(x) - x_min) / (x_max - x_min)
    norm_y = (np.array(y) - y_min) / (y_max - y_min)
    trans_x = (norm_x * bbox_ax.width ) + bbox_ax.x0
    trans_y = (norm_y * bbox_ax.height) + bbox_ax.y0

    # Add 1st and last points.
    x0, y0 = extrapolate(trans_x[:+2], trans_y[:+2], False)
    x1, y1 = extrapolate(trans_x[-2:], trans_y[-2:], True )
    nx = np.r_[x0, trans_x, x1]
    ny = np.r_[y0, trans_y, y1]
    
    vectors, degrees = [], []
    quad_left, quad_right = [], []
    for n in np.arange(1,len(nx)-1): 
        
        # Center coordinates i.e. (0,0)
        x0, x1 = nx[[n-1, n+1]] - nx[n]
        y0, y1 = ny[[n-1, n+1]] - ny[n]
        
        # Unit vector.
        mag0 = np.sqrt(pow(x0,2) + pow(y0,2))
        mag1 = np.sqrt(pow(x1,2) + pow(y1,2))
        y0, y1 =  y0 / mag0, y1 / mag1
        x0, x1 =  x0 / mag0, x1 / mag1
        
        # Determine angles (radian)
        rad_left  = np.arctan2(y0, x0)
        rad_right = np.arctan2(y1, x1)
        rad = rad_left - rad_right 
        if rad < 0: rad = 2*np.pi + rad
            
        rad *= 0.5
        r = np.array([[np.cos(rad),-np.sin(rad)],
                      [np.sin(rad), np.cos(rad)]])
        x2, y2 = r.dot(np.r_[[x1, y1]].reshape(-1,1)).ravel()
        
        # Scale to unit vector.
        len2 = np.sqrt(pow(x2,2) + pow(y2,2))
        vectors += [(x2/len2, y2/len2)]
        rad2 = np.arctan2(y2, x2)
        if rad2 < 0: rad2 = 2*np.pi + rad2
        degrees += [rad2/np.pi*180]
        quad_left  += [find_quadrant(x0, y0)]
        quad_right += [find_quadrant(x1, y1)]

    return vectors, degrees, quad_left, quad_right

def plot_elbow_base(score, ax=None, color="#FC427B", plot_kwds=None, 
                    show_anno=True, anno_format=None, tight_layout=True):
    
    '''
    Plot Elbow-index.

    Parameters
    ----------
    score : tuple of arrays
        Output from "elbow_index" function.

    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, ax is created with 
        default figsize.
        
    color : Color-hex, default="#FC427B"
        Color to be passed to "ax.plot". This overrides "plot_kwds". 
        
    plot_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.plot".
    
    show_anno : bool, default=True
        If True, it annotates the point xy with y.
    
    anno_format : string formatter, default=None
        String formatters (function) for ax.annotate values. If None, 
        it defaults to "{:,.2f}".format.
    
    tight_layout : bool, default=True
        If True, it adjusts the padding between and around subplots 
        i.e. plt.tight_layout().
    
    Returns
    -------
    ax : Matplotlib axis object
        
    '''
    ax = get_ax(ax, score[1])
    x  = np.arange(len(score[1]))
    kwds = dict(linewidth=5, solid_capstyle='round', 
                solid_joinstyle="round")
    if plot_kwds is not None: kwds.update(plot_kwds)
    kwds.update({"color":color})
    ax.plot(x, score[0], **kwds)
    
    if show_anno: set_annotate(ax, x, score[0], color, anno_format)
    set_yaxis(ax, "Score", r"$\gamma(k)$")
    xlabel = r'Elbow : Choose $k$ that maximizes $\gamma(k)$'
    set_xaxis(ax, x, score[1], xlabel, "$k$")
    set_axvline(ax, np.argmax(score[0]), "$k$*", color)
    if tight_layout: plt.tight_layout()
        
    return ax

def plot_wb_base(score, ax=None, color="#FC427B", plot_kwds=None, 
                 show_anno=True, anno_format=None, tight_layout=True):
    
    '''
    Plot WB-index.
    
    Parameters
    ----------
    score : tuple of arrays
        Output from "wb_index" function.
    
    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, ax is created with 
        default figsize.
    
     color : Color-hex, default="#FC427B"
        Color to be passed to "ax.plot". This overrides "plot_kwds".
        
    plot_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.plot".
    
    show_anno : bool, default=True
        If True, it annotates the point xy with y.
    
    anno_format : string formatter, default=None
        String formatters (function) for ax.annotate values. If None, 
        it defaults to "{:,.2f}".format.
    
    tight_layout : bool, default=True
        If True, it adjusts the padding between and around subplots 
        i.e. plt.tight_layout().
    
    Returns
    -------
    ax : Matplotlib axis object
    
    '''
    ax = get_ax(ax, score[1])    
    x = np.arange(len(score[1]))
    xticklabels = np.array(score[1]).astype(int)
    
    kwds = dict(linewidth=5, solid_capstyle='round', 
                solid_joinstyle="round")
    if plot_kwds is not None: kwds.update(plot_kwds)
    kwds.update({"color":color})
    ax.plot(x, score[0], **kwds)
    
    if show_anno: set_annotate(ax, x, score[0], color, anno_format)
    set_yaxis(ax, "Score", r"$WB(k)$")
    xlabel = r'WB : Choose $k$ that minimizes $WB(k)$'
    set_xaxis(ax, x, xticklabels, xlabel, "$k$")
    
    # Select minimum score, where k>1
    adj_score = np.where(xticklabels==1, np.inf ,score[0])
    set_axvline(ax, np.argmin(adj_score), "$k$*", color)
    if tight_layout: plt.tight_layout()

    return ax

def plot_ch_base(score, ax=None, color="#FC427B", plot_kwds=None, 
                 show_anno=True, anno_format=None, tight_layout=True):
    
    '''
    Plot Calinski-Harabasz index.
    
    Parameters
    ----------
    score : tuple of arrays
        Output from "calinski_harabasz" function.
    
    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, ax is created with 
        default figsize.
    
    color : Color-hex, default="#FC427B"
        Color to be passed to "ax.plot". This overrides "plot_kwds".
        
    plot_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.plot".
    
    show_anno : bool, default=True
        If True, it annotates the point xy with y.
    
    anno_format : string formatter, default=None
        String formatters (function) for ax.annotate values. If None, 
        it defaults to "{:,.2f}".format.
    
    tight_layout : bool, default=True
        If True, it adjusts the padding between and around subplots 
        i.e. plt.tight_layout().
    
    Returns
    -------
    ax : Matplotlib axis object
    
    '''
    ax = get_ax(ax, score[1])   
    x  = np.arange(len(score[1]))
    xticklabels = np.array(score[1]).astype(int)
    
    kwds = dict(linewidth=5, solid_capstyle='round', 
                solid_joinstyle="round")
    if plot_kwds is not None: kwds.update(plot_kwds)
    kwds.update({"color":color})
    amin, amax = np.percentile(score[0], q=[0,100])       
    normal = (score[0] - amin)/(amax - amin)  
    ax.plot(x, normal, **kwds)
    
    if show_anno: set_annotate(ax, x, normal, color, anno_format)
    set_yaxis(ax, "Normalized Score", r"$CH(k)$")
    xlabel = r'Calinski Harabasz : Choose $k$ that maximizes $CH(k)$'
    set_xaxis(ax, x, xticklabels, xlabel, "$k$")
    
    # Select maximum score, where k>1
    adj_score = np.where(xticklabels==1, 0, normal)
    set_axvline(ax, np.argmax(adj_score), "$k$*", color)
    if tight_layout: plt.tight_layout()

    return ax

def plot_hartigan_base(score, ax=None, color="#FC427B", plot_kwds=None, 
                       show_anno=True, anno_format=None, tight_layout=True):
    
    '''
    Plot Hartigan-index (modified). We choose the smallest k whose 
    H(k) is below the average of all H(k).
    
    Parameters
    ----------
    score : tuple of arrays
        Output from "hartigan_index" function.
    
    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, ax is created with 
        default figsize.
    
    color : Color-hex, default="#FC427B"
        Color to be passed to "ax.plot". This overrides "plot_kwds".
        
    plot_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.plot".
    
    show_anno : bool, default=True
        If True, it annotates the point xy with y.
    
    anno_format : string formatter, default=None
        String formatters (function) for ax.annotate values. If None, 
        it defaults to "{:,.2f}".format.
    
    tight_layout : bool, default=True
        If True, it adjusts the padding between and around subplots 
        i.e. plt.tight_layout().
    
    Returns
    -------
    ax : Matplotlib axis object
    
    '''
    ax = get_ax(ax, score[1])
    x  = np.arange(len(score[1]))
    xticklabels = np.array(score[1]).astype(int)
    
    kwds = dict(linewidth=5, solid_capstyle='round', 
                solid_joinstyle="round")
    if plot_kwds is not None: kwds.update(plot_kwds)
    kwds.update({"color":color})
    amin, amax = np.percentile(score[0], q=[0,100])       
    normal = (score[0] - amin)/(amax - amin)   
    ax.plot(x, normal, **kwds)
    
    if show_anno: set_annotate(ax, x, normal, color, anno_format)
    set_yaxis(ax, "Normalized Score", r"$H(k)$")
    xlabel = r'Hartigan : Choose the smallest $k$ such that $H(k) \leq \eta$'
    set_xaxis(ax, x, xticklabels, xlabel, "$k$")
    
    mean = np.mean(normal)
    ax.axhline(mean, color="#cccccc", lw=0.8, zorder=-1)
    args = (ax.transAxes, ax.transData)
    trans = transforms.blended_transform_factory(*args)
    ax.text(1.01, mean, r"$\eta$ =" + "\n{:.2f}".format(mean), 
            transform=trans, fontsize=13, va="center", ha="left")
    
    y_locator = FixedLocator([mean])
    ax.yaxis.set_minor_locator(y_locator)
    ax.tick_params(axis="y", which="minor", length=3, color="k")
    
    # Select minimum score, where k>1
    adj_score = np.where(xticklabels==1, np.inf, normal)
    set_axvline(ax, np.argmax(adj_score<=mean), "$k$*", color)
    if tight_layout: plt.tight_layout()
    
    return ax

def plot_silhouette_base(score, ax=None, color="#FC427B", plot_kwds=None, 
                         show_anno=True, anno_format=None, tight_layout=True):
    
    '''
    Plot Silhouette-score.
    
    Parameters
    ----------
    score : tuple of arrays
        Output from "silhouette" function.
    
    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, ax is created with 
        default figsize.
    
    color : Color-hex, default="#FC427B"
        Color to be passed to "ax.plot". This overrides "plot_kwds".
        
    plot_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.plot".
    
    show_anno : bool, default=True
        If True, it annotates the point xy with y.
    
    anno_format : string formatter, default=None
        String formatters (function) for ax.annotate values. If None, 
        it defaults to "{:,.2f}".format.
    
    tight_layout : bool, default=True
        If True, it adjusts the padding between and around subplots 
        i.e. plt.tight_layout().
    
    Returns
    -------
    ax : Matplotlib axis object
    
    '''
    ax = get_ax(ax, score[1]) 
    x  = np.arange(len(score[1]))
    xticklabels = np.array(score[1]).astype(int)
    
    kwds = dict(linewidth=5, solid_capstyle='round', 
                solid_joinstyle="round")
    if plot_kwds is not None: kwds.update(plot_kwds)
    kwds.update({"color":color})
    ax.plot(x, score[0], **kwds)
    
    if show_anno: set_annotate(ax, x, score[0], color, anno_format)
    set_yaxis(ax, "Score", r"$sil(k)$")
    xlabel = (r'Silhouette : Choose $k$ that maximizes $sil(k)$',
              r'(Artificial : $\geq$0.25,'
              r' Reasonable : $\geq$0.50,'
              r' Strong : $\geq$0.70)')
    set_xaxis(ax, x, xticklabels, "\n".join(xlabel), "$k$")
    
    y_locator = FixedLocator([0.25,0.5,0.7])
    ax.yaxis.set_minor_locator(y_locator)
    ax.tick_params(axis="y", which="minor", length=3, color="k")

    y_min, y_max = ax.get_ylim()
    args = (ax.transAxes, ax.transData)
    trans = transforms.blended_transform_factory(*args)
    for v in [0.25,0.5,0.7]:
        if (y_min<=v) & (v<=y_max):
            ax.axhline(v, color="#cccccc", lw=0.8, zorder=-1)
            ax.text(1.01, v, "{:.2f}".format(v), transform=trans, 
                    fontsize=12, va="center", ha="left", 
                    zorder=-1, color="#cccccc")
    
    # Select maximum score, where k>1
    adj_score = np.where(xticklabels==1, -np.inf, score[0])
    set_axvline(ax, np.argmax(adj_score), "$k$*", color)
    if tight_layout: plt.tight_layout()
  
    return ax  

def plot_gap1_base(score, ax=None, color="#FC427B", plot_kwds=None, 
                   show_anno=True, anno_format=None, tight_layout=True):
    
    '''
    Plot Gap-Statistics of all `k` group i.e. Gap(k) = E{Log(W(kb))} 
    - Log(W(k)).
    
    Parameters
    ----------
    score : tuple of arrays
        Output from "gap_statistics" function.

    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, ax is created with 
        default figsize.
    
    color : Color-hex, default="#FC427B"
        Color to be passed to "ax.plot". This overrides "plot_kwds".
        
    plot_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.plot".

    show_anno : bool, default=True
        If True, it annotates the point xy with y.
    
    anno_format : string formatter, default=None
        String formatters (function) for ax.annotate values. If None, 
        it defaults to "{:,.2f}".format.
    
    tight_layout : bool, default=True
        If True, it adjusts the padding between and around subplots 
        i.e. plt.tight_layout().
        
    Returns
    -------
    ax : Matplotlib axis object
    
    '''
    ax = get_ax(ax, score[5])
    x  = np.arange(len(score[5]))
    xticklabels = np.array(score[5]).astype(int)
    
    kwds = dict(linewidth=5, solid_capstyle='round', 
                solid_joinstyle="round")
    if plot_kwds is not None: kwds.update(plot_kwds)
    kwds.update({"color": color})

    # Gap(k) = E{Log(W(kb))} - Log(W(k))
    ax.plot(x, score[3], **kwds)
    
    if show_anno: set_annotate(ax, x, score[3], color, anno_format)
    # r"$E\{\log{(W_{kb})}\}-\log{(W_{k})}$"
    set_yaxis(ax, "Score", r"$Gap(k)$")
    xlabel = r'Gap-Statistics : Choose $k$ that maximizes $Gap(k)$'
    set_xaxis(ax, x, xticklabels, xlabel, "$k$")
    
    # Select maximum score, where k>1
    adj_score = np.where(xticklabels==1, 0, abs(score[3]))
    set_axvline(ax, np.argmax(adj_score), "$k$*", color)
    if tight_layout: plt.tight_layout()
        
    return ax

def plot_gap2_base(score, ax=None, color="#FC427B", plot_kwds=None, 
                   show_anno=True, anno_format=None, tight_layout=True):
    
    '''
    Plot Delta(k) = Gap(k) - Gap(k+1) + s(k+1).
    
    Parameters
    ----------
    score : tuple of arrays
        Output from "gap_statistics" function.

    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, ax is created with 
        default figsize.
    
    color : Color-hex, default="#FC427B"
        Color to be passed to "ax.plot". This overrides "plot_kwds".
        
    plot_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.plot".

    show_anno : bool, default=True
        If True, it annotates the point xy with y.
    
    anno_format : string formatter, default=None
        String formatters (function) for ax.annotate values. If None, 
        it defaults to "{:,.2f}".format.
    
    tight_layout : bool, default=True
        If True, it adjusts the padding between and around subplots 
        i.e. plt.tight_layout().
        
    Returns
    -------
    ax : Matplotlib axis object
    
    '''
    ax = get_ax(ax, score[5])
    x  = np.arange(len(score[5]))
    xticklabels = np.array(score[5]).astype(int)
    
    kwds = dict(linewidth=5, solid_capstyle='round', 
                solid_joinstyle="round")
    if plot_kwds is not None: kwds.update(plot_kwds)
    kwds.update({"color": color})
     
    # Delta(k) = Gap(k) - Gap(k+1) + s_{k+1}
    ax.plot(x, score[4], **kwds)
    
    if show_anno: set_annotate(ax, x, score[4], color, anno_format)
    # r"$Gap(k)-Gap(k+1)+s_{k+1}$"
    set_yaxis(ax, "Score", r"$\Delta(k)$")
    xlabel = (r'Gap-Statistics : Choose the smallest '
              r'$k$ such that $\Delta \geq0$')
    set_xaxis(ax, x, xticklabels, xlabel, "$k$")
    
    y_min, y_max = ax.get_ylim()
    if (y_min<=0) & (0<=y_max):
        ax.axhline(0, color="#cccccc", lw=0.8, zorder=-1)
        
    # Select maximum score, where k>1
    adj_score = np.where(xticklabels==1, -np.inf, score[4])
    set_axvline(ax, np.argmax(adj_score), "$k$*", color)
    if tight_layout: plt.tight_layout()
        
    return ax

def plot_dh_base(score, ax=None, color="#FC427B", plot_kwds=None, 
                 show_anno=True, anno_format=None, tight_layout=True):
    
    '''
    Plot Duda and Hart (1973) index.
    
    Parameters
    ----------
    score : tuple of arrays
        Output from "dudahart_index" function.
    
    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, ax is created with 
        default figsize.
    
    color : Color-hex, default="#FC427B"
        Color to be passed to "ax.plot". This overrides "plot_kwds".
        
    plot_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.plot".
    
    show_anno : bool, default=True
        If True, it annotates the point xy with y.
    
    anno_format : string formatter, default=None
        String formatters (function) for ax.annotate values. If None, 
        it defaults to "{:,.2f}".format.
    
    tight_layout : bool, default=True
        If True, it adjusts the padding between and around subplots 
        i.e. plt.tight_layout().
    
    Returns
    -------
    ax : Matplotlib axis object
    
    '''
    ax = get_ax(ax, score[1])    
    x  = np.arange(len(score[1]))
    xticklabels = np.array(score[1]).astype(int)
    t = score[2]
    
    kwds = dict(linewidth=5, solid_capstyle='round', 
                solid_joinstyle="round")
    if plot_kwds is not None: kwds.update(plot_kwds)
    kwds.update({"color":color})
    ax.plot(x, score[0], **kwds)
    
    if show_anno: set_annotate(ax, x, score[0], color, anno_format)
    set_yaxis(ax, "Score", r"$DH(k)$")
    set_xaxis(ax, x, xticklabels, 
              (r'Duda-Hart : Choose the smallest '
               r'$k$ such that $DH(k) \geq DH^{*}$'), "$k$")
    
    y_min, y_max = ax.get_ylim()
    if (y_min<=t) & (t<=y_max):
        ax.axhline(t, color="#cccccc", lw=0.8, zorder=-1)
        args  = (ax.transAxes, ax.transData)
        trans = transforms.blended_transform_factory(*args)
        ax.text(1.01, t, r"$DH^{*}$ = " + "\n{:.2f}".format(t), 
                transform=trans, fontsize=13, va="center", ha="left")
        y_locator = FixedLocator([t])
        ax.yaxis.set_minor_locator(y_locator)
        ax.tick_params(axis="y", which="minor", length=3, color="k")  
    
    # Select maximum score, where k>1
    adj_score = np.where(xticklabels==1, -np.inf, score[0])
    set_axvline(ax, np.argmax(adj_score>=t), "$k$*", color)
    if tight_layout: plt.tight_layout()

    return ax