import numpy as np
import shared_dataset
import visualize
def K_means_init(X, K):
    # initialize sample, weight, and cov
    n, feature = X.shape
    index = np.random.choice(n, K, replace=False)
    means =  np.array(X[index])
    # print("shape:", means.shape, covs.shape, weight.shape)
    return means
def distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2)) 
def calculate_centroid(X, K, pred_label):
    N, feature = X.shape
    centroid = np.zeros((K, feature))
    for i in range(K):
        points = X[pred_label == i]
        if len(points) > 0:
            centroid[i] = np.mean(points, axis=0)
    return centroid
def assign_cluster(X, centroids):
    pred_label = []
    for point in X:
        distantces = [distance(point, ct) for ct in centroids]
        pred_label.append(np.argmin(distantces))
    return np.array(pred_label)
def K_means(X, K, max_iter):
    cent = K_means_init(X, K)
    for t in range(max_iter):
        pred_label = assign_cluster(X, cent)
        new_cent = calculate_centroid(X, K, pred_label)
        if np.all(new_cent == cent):
            break
        cent = new_cent
    return pred_label, cent
    
visualize.plot(shared_dataset.y, shared_dataset.X, 3, 'a')
predicted_label, initial_means = K_means(shared_dataset.X, 2, 100)
predicted_label += 1
visualize.plot(predicted_label, shared_dataset.X, 2, 'a', initial_means)
predicted_label, initial_means = K_means(shared_dataset.X, 3, 100)
predicted_label += 1
visualize.plot(predicted_label, shared_dataset.X, 3, 'a', initial_means)
predicted_label, initial_means = K_means(shared_dataset.X, 4, 100)
predicted_label += 1
visualize.plot(predicted_label, shared_dataset.X, 4, 'a', initial_means)


visualize.plot(shared_dataset.y, shared_dataset.X_b, 3, 'b')
predicted_label, initial_means = K_means(shared_dataset.X_b, 2, 100)
predicted_label += 1
visualize.plot(predicted_label, shared_dataset.X_b, 2, 'b', initial_means)
predicted_label, initial_means = K_means(shared_dataset.X_b, 3, 100)
predicted_label += 1
visualize.plot(predicted_label, shared_dataset.X_b, 3, 'b', initial_means)
predicted_label, initial_means = K_means(shared_dataset.X_b, 4, 100)
predicted_label += 1
visualize.plot(predicted_label, shared_dataset.X_b, 4, 'b', initial_means)
