import matplotlib.pyplot as plt
def plot(pre_label, X, K, ex, means_initial=None):
    colors = ['r', 'g', 'b', 'c']
    
    # 創建一個新的圖形
    plt.figure(figsize=(8, 6))
    
    # 根據 predicted_label 的分類結果為每個數據點選擇顏色
    for i in range(1, K + 1):
        plt.scatter(X[pre_label == i, 0], X[pre_label == i, 1], 
                    label=f'Class {i}', color=colors[i-1], s=10)
    if means_initial is not None:
        plt.scatter(means_initial[:, 0], means_initial[:, 1], 
                    color='black', label='Initial Means', marker='x', s=100, edgecolors='white')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'EM Algorithm Classification Results, from ({ex})')
    plt.show()