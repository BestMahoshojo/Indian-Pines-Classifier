import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_indian_pines_data():
    """
    加载 Indian Pines 数据集
    返回: X (特征), y (标签), image_shape (图像形状)
    """
    # 注意: 实际使用时需要从文件加载真实数据
    # 这里创建模拟数据用于演示
    print("正在加载 Indian Pines 数据集...")
    
    # 模拟 Indian Pines 数据 (145x145 像素, 200 个波段)
    height, width, bands = 145, 145, 200
    image_shape = (height, width)
    
    # 创建模拟高光谱数据
    np.random.seed(42)
    X = np.random.randn(height * width, bands) * 100 + 500
    X = np.abs(X)  # 确保值为正
    
    # 创建模拟地面真实标签 (16类)
    # 实际应从 .mat 文件加载真实标签
    y = np.zeros(height * width, dtype=np.int8)
    
    # 模拟一些类别区域
    class_positions = [
        (20, 30, 40, 50, 1),   # 区域1: 类别1
        (60, 80, 30, 60, 2),   # 区域2: 类别2
        (100, 120, 70, 100, 3), # 区域3: 类别3
        (10, 40, 100, 130, 4), # 区域4: 类别4
        (80, 100, 10, 40, 5),  # 区域5: 类别5
    ]
    
    for y1, y2, x1, x2, class_id in class_positions:
        for i in range(y1, y2):
            for j in range(x1, x2):
                idx = i * width + j
                y[idx] = class_id
    
    # 添加一些随机噪声类别
    random_indices = np.random.choice(height * width, size=500, replace=False)
    y[random_indices] = np.random.randint(6, 16, size=500)
    
    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    print(f"数据加载完成: X形状={X.shape}, y形状={y.shape}, 图像形状={image_shape}")
    return X, y, image_shape

def prepare_for_classification(X, y, test_size=0.3, random_state=42):
    """为监督分类准备训练和测试数据"""
    from sklearn.model_selection import train_test_split
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"数据划分: 训练集 {X_train.shape}, 测试集 {X_test.shape}")
    return X_train, X_test, y_train, y_test