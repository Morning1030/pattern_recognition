import numpy as np
def bayesian(x, means, covs, pr):
    predict = []
    inv_covs = [np.linalg.inv(cov) for cov in covs]
    log_det_covs = [np.log(np.linalg.det(cov)) for cov in covs]
    for data in x:
        g = []
        for mean, inv_cov, log_det_cov, p in zip(means, inv_covs, log_det_covs, pr):
            g.append(-0.5 * ((data - mean) @ inv_cov @ (data - mean) + log_det_cov) + np.log(p))
        predict.append(np.argmax(g))
    return predict

def euclidean(x, means):
    predict = []
    for data in x:
        # calculate the distance
        distance = []
        for i, mean in enumerate(means):
            distance.append(np.sqrt(np.sum(np.square(data - mean))))
        predict.append(np.argmin(distance))
    return predict

def mahalanobis(x, means, covs):
    predict = []
    inv_covs = []
    # calculate inverse matrices of cov
    for cov in covs:
        inv_covs.append(np.linalg.inv(cov))
    for data in x:
        distance = []
        for mean, inv_cov, in zip(means, inv_covs):
            distance.append(np.sqrt((data - mean).T @ inv_cov @ (data - mean)))
        predict.append(np.argmin(distance))   
        # calculate the inverse matrix of cov
    return predict

def compute_error(predict, y):
   return np.mean(predict != y) 