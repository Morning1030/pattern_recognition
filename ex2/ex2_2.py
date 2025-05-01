import numpy as np

def generate_dataset(N, p):
    x = np.random.binomial(1, p, N)
    return x
def p_ml_estimator(dataset):
    p_ml = np.mean(dataset)
    return p_ml 
a = generate_dataset(1000, 0.7)
b = generate_dataset(5000, 0.7)
p_a = p_ml_estimator(a)
p_b = p_ml_estimator(b)

print("p_ml_a", p_a)
print("p_ml_b", p_b)