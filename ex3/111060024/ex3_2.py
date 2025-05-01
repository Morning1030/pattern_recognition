# import library
import numpy as np
import matplotlib.pyplot as plt
import shared_dataset
import visualize

def gaussian_pdf(X, mean, cov):
    # print("cov", cov)
    feature = X.shape[0]
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    X_center = X - mean
    norm_constant = 1.0 / (np.sqrt((2 * np.pi) ** feature * det))
    # exponent = 1.0 / np.sqrt(det)
    exponent = np.exp(-0.5 * (X_center).T @ inv @ X_center)
    return norm_constant * exponent
    
def EM_initialize(X, K):
    # initialize sample, weight, and cov
    n, feature = X.shape
    index = np.random.choice(n, K, replace=False)
    means =  np.array(X[index])
    print("in EM_initialize:", index)
    # try to choose better start
    covs = np.array([np.eye(feature) for i in range(K)])
    weight = np.ones(K) / K
    # print("shape:", means.shape, covs.shape, weight.shape)
    return means, covs, weight
    
def EM_algorithm(X, K, max_iter, epsilon=1e-4):
    # Initialization
    means, covs, weight = EM_initialize(X, K)
    initial_means = means.copy()
    N, feature = X.shape
    for t in range(max_iter):
        # backup t'th iteration
        mean_old = means.copy()
        covs_old = covs.copy()
        weight_old = weight.copy()
        
        gammas = np.zeros((N, K))
        # E step
        for j in range(K):
            for k in range(N):
                gammas[k, j] = weight[j] * gaussian_pdf(X[k], means[j], covs[j])
        gammas = gammas / np.sum(gammas, axis=1, keepdims=True) 
        
        N_k = np.sum(gammas, axis=0)        
        # M step
        for j in range(K):
            # update mean
            means[j] = np.sum(gammas[:, j].reshape(-1, 1) * X, axis=0) / N_k[j]
            # update cov
            X_center = X - means[j]
            covs[j] = np.zeros((feature, feature))
            for i in range(N):
                covs[j] += gammas[i, j] * (X_center[i, :, np.newaxis] @ X_center[i, np.newaxis, :])
            covs[j] /= N_k[j]
            # covs[j] = np.sum((gammas[:, j].reshape(-1, 1) * (X_center[:, :, np.newaxis] @ X_center[:, np.newaxis, :])), axis=0) / N_k[j]
            # update weight
            weight[j] = N_k[j] / N
            
        diff_mean = np.linalg.norm(means - mean_old)
        diff_cov = np.linalg.norm(covs - covs_old)
        diff_weight = np.linalg.norm(weight - weight_old)    
        if ((diff_mean + diff_cov + diff_weight) < epsilon):
            break
    return gammas, initial_means


    
Gamma, initial_means = EM_algorithm(shared_dataset.X, 3, 100)
predicted_label = np.argmax(Gamma, axis=1) + 1

visualize.plot(shared_dataset.y, shared_dataset.X, 3, 'a')
visualize.plot(predicted_label, shared_dataset.X, 3, 'a', initial_means)


Gamma_b, initial_means_b = EM_algorithm(shared_dataset.X_b, 3, 100)
predicted_label_b = np.argmax(Gamma_b, axis=1) + 1

visualize.plot(shared_dataset.y, shared_dataset.X_b, 3, 'b')
visualize.plot(predicted_label_b, shared_dataset.X_b, 3, 'b', initial_means_b)
