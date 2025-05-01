import numpy as np
def generate_dataset(N, means, covs, pr):
    x = []
    for mean, cov, p in zip(means, covs, pr):
        x.append(np.random.multivariate_normal(mean, cov, int(N * p)))
    x = np.vstack(x)
    return x
def generate_labelset(N, pr):
    y = []
    for i, p in enumerate(pr):
        y.append(np.full(int(N * p), i))
    y = np.concatenate(y)
    return y