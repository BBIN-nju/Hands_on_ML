import numpy as np
import matplotlib.pyplot as plt
import os

def distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

class KNN:
    def __init__(self, k, label_num):
        self.k = k
        self.label_num = label_num # 类别数量
        
    def fit(self, x_train, y_train):
        # 在类中保存训练数据
        self.x_train = x_train
        self.y_train = y_train
    
    def get_knn_indices(self, x):
        # 获取距离目标样本x最近的k个样本的索引
        # 计算已知样本到目标样本的距离
        dis = list(map(lambda a: distance(a, x), self.x_train))
        # 按距离从小到大排序，并得到对应的下表
        knn_indices = np.argsort(dis)
        # 获取最近K个下表
        knn_indices = knn_indices[:self.k]
        return knn_indices
    
    def get_label(self, x):
        # 对KNN方法的具体实现，观察k个近邻并使用np.argmax获取其中数量最多的类别
        knn_indices = self.get_knn_indices(x)
        # 类别计数
        label_statistic = np.zeros(shape=[self.label_num])
        for idx in knn_indices:
            label = int(self.y_train[idx])
            label_statistic[label] += 1
        # 返回数量最多的类别
        return np.argmax(label_statistic)
    
    def predict(self, x_test):
        # 预测样本 test_x 的类别
        predicted_test_labels = np.zeros(shape=[len(x_test)], dtype=int)
        for i, x in enumerate(x_test):
            predicted_test_labels[i] = self.get_label(x)
        return predicted_test_labels


dataset_root = "..\Dataset"

"""
train-images-idx3-ubyte 60000 张图片，大小为 60000 * 28 * 28 + 16, 有16B magic用于存储文件信息
train-labels-idx1-ubyte 60000 个标签，有 8B magic 用于存储文件信息
"""
# 读入 MNIST 数据集
data_dir = os.path.join(dataset_root, "MNIST") # MNIST 数据集所在的文件夹
fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte')) 
loaded = np.fromfile(file=fd , dtype=np.uint8)
m_x = loaded[16:].reshape(60000, 28, 28)

fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte')) 
loaded = np.fromfile(file=fd , dtype=np.uint8)
m_y = loaded[8:]
print("dataset size: ",m_x.shape) # (60000 * 28 * 28,) 此时读出来还是一维向量
print("dataset label size",m_y.shape) # (60000,) 一维向量

# 将数据集划分为训练集和测试集
ratio = 0.8
split = int(len(m_x) * ratio)

# 打乱数据
np.random.seed(0)
idx = np.random.permutation(np.arange(len(m_x))) # 从 1~60000 的一个排列中随机生成一个
m_x = m_x[idx]
m_y = m_y[idx]
x_train, x_test = m_x[:split], m_x[split:]
y_train, y_test = m_y[:split], m_y[split:]

for k in range(1, 10):
    knn = KNN(k, label_num=10)
    knn.fit(x_train, y_train)
    predicted_labels = knn.predict(x_test)
    accuracy = np.mean(predicted_labels == y_test)
    print(f'K的取值为 {k}, 预测准确率为 {accuracy * 100 :.1f}%')