import numpy as np
import scipy.stats
from scipy.spatial.distance import cdist


# FUNCTION TO CALCULATE LOG LIKELIHOOD
def calc_loglikelihood(x, pis, means, cov, mix):
    N = x.shape[0]
    d = x.shape[1]
    ll = np.array([pis[j] * scipy.stats.multivariate_normal(means[j], cov[j], allow_singular=True).pdf(x) for j in range(mix)])
    ll_sum = np.sum(ll, axis=0)
    log_ll = np.log(ll_sum)
    log_ll_sum = (1/N)*np.sum(log_ll)
    return log_ll_sum

# FUNCTION TO CALCULATE RESPONSIBILITY
def calc_responsibility(x, pis, means, cov, mix):
    N = x.shape[0]
    d = x.shape[1]
    resp = np.array([pis[j] * scipy.stats.multivariate_normal(means[j], cov[j], allow_singular=True).pdf(x) for j in range(mix)]).T
    resp = resp.reshape((N, mix))
    resp_sum = np.sum(resp, axis=1)
    resp_sum = resp_sum.reshape((N, 1))
    resp = resp/resp_sum
    return resp

# FUNCTION TO CALCULATE Nk
def calc_Nk(resp, mix):
    Nk = np.sum(resp, axis=0)
    Nk = Nk.reshape((mix, 1))
    return Nk

# FUNCTION TO UPDATE THE WEIGHTS / PRIOR PROBABILITIES
def update_weights(Nk, N):
    pis = Nk/N
    return pis

# FUNCTION TO UPDATE THE MEANS
def update_means(x, resp, Nk):
    means = (np.dot(resp.T, x))/Nk
    return means

# FUNCTION TO UPDATE THE COVARIANCE MATRIX
def update_cov_matrix(x, resp, means, Nk, cov_type, mix):
    N = x.shape[0]
    d = x.shape[1]
    cov = [np.sum(np.array([resp[i][j] * np.dot((x[i] - means[j]).reshape((d, 1)), (x[i] - means[j]).reshape((d, 1)).T) for i in range(N)]), axis=0) for j in range(mix)]
    cov = np.array(cov)
    if(cov_type == 'diag'):
        cov = np.array([np.diag(np.diag(cov[i])) for i in range(mix)])
    cov = np.array([cov[i]/Nk[i] for i in range(mix)])
    return cov


# FUNCTION TO FIND K-MEANS CLUSTER CENTROIDS AND LABELS FOR EACH SAMPLE
def kmeans(x, mix, itr):
    N = x.shape[0]
    idx = np.random.choice(N, mix)
    centroids = x[idx, :]
    distances = cdist(x, centroids ,'euclidean')
    labels = np.array([np.argmin(i) for i in distances])
    for _ in range(itr):
        centroids = []
        for idx in range(mix):
            temp = x[labels==idx].mean(axis=0) 
            centroids.append(temp)
        centroids = np.vstack(centroids)
        distances = cdist(x, centroids ,'euclidean')
        labels = np.array([np.argmin(i) for i in distances])
    return centroids, labels


# FUNCTION FOR OBTAINING GMM PARAMETERS/ EM ALGORITHM
def gmm(x, itr, mix, cov_type):
    N = x.shape[0]
    d = x.shape[1]
    # K-MEANS INITIALIZATION
    means, labels = kmeans(x, mix, 50)
    
    # INITIALIZE WEIGHTS, MEANS, AND COVARIANCE MATRICES
    cov = []
    pis = []
    
    for i in range(mix):
        dat = x[labels == i].T
        p = dat.shape[1]/N
        dat = (1/p) * np.cov(dat)
        cov.append(dat)
        pis.append(p)
    
    cov = np.array(cov)
    pis = np.array(pis)
    
    # RUN EM-ALGORITHM FOR GIVEN NO. OF ITERATIONS
    log_likelihood = []
    for itr in range(itr):
        # E - STEP
        resp = calc_responsibility(x, pis, means, cov, mix)
        # M - STEP
        Nk = calc_Nk(resp)
        pis = update_weights(Nk, N)
        means = update_means(x, resp, Nk)
        cov = update_cov_matrix(x, resp, means, Nk, cov_type, mix)
        log_likelihood.append(calc_loglikelihood(x, pis, means, cov, mix))
        print('Log Likelihood = ' + str(log_likelihood[-1]))

    return pis, means, cov, log_likelihood

