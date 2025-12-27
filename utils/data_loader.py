import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
import os
warnings.filterwarnings('ignore')

# 设置matplotlib后端
import matplotlib
matplotlib.use('Agg')

def load_indian_pines_data(use_real_data=False):
    """
    加载 Indian Pines 数据集
    
    参数:
    - use_real_data: 是否使用真实数据（需要下载.mat文件）
    
    返回: X (特征), y (标签), image_shape (图像形状)
    """
    print("正在加载 Indian Pines 数据集...")
    
    if use_real_data:
        # 尝试加载真实数据
        try:
            from scipy.io import loadmat
            
            # 真实数据文件路径
            data_dir = 'data/Indian_pines'
            data_file = os.path.join(data_dir, 'Indian_pines_corrected.mat')
            label_file = os.path.join(data_dir, 'Indian_pines_gt.mat')
            
            if not os.path.exists(data_file) or not os.path.exists(label_file):
                print("未找到真实数据文件，使用模拟数据")
                return load_simulated_data()
            
            # 加载真实数据
            data = loadmat(data_file)
            label_data = loadmat(label_file)
            
            # 提取数据 - 不同版本的文件可能有不同的字段名
            if 'indian_pines_corrected' in data:
                X_3d = data['indian_pines_corrected']
            elif 'indian_pines' in data:
                X_3d = data['indian_pines']
            else:
                # 尝试获取第一个非元数据的数组
                for key in data.keys():
                    if not key.startswith('__') and data[key].ndim == 3:
                        X_3d = data[key]
                        break
            
            # 提取标签
            if 'indian_pines_gt' in label_data:
                y_2d = label_data['indian_pines_gt']
            elif 'gt' in label_data:
                y_2d = label_data['gt']
            else:
                for key in label_data.keys():
                    if not key.startswith('__') and label_data[key].ndim == 2:
                        y_2d = label_data[key]
                        break
            
            # 获取图像形状
            height, width, bands = X_3d.shape
            image_shape = (height, width)
            
            # 重塑为二维数组（样本数 × 特征数）
            X = X_3d.reshape(-1, bands)
            y = y_2d.reshape(-1)
            
            # 移除标签为0的背景像素（可选）
            # mask = y != 0
            # X = X[mask]
            # y = y[mask]
            
            print(f"真实数据加载完成: X形状={X.shape}, y形状={y.shape}, 图像形状={image_shape}")
            print(f"类别标签: {np.unique(y)}")
            
            # 数据标准化
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            
            return X, y, image_shape
            
        except Exception as e:
            print(f"加载真实数据失败: {str(e)}，使用模拟数据")
            return load_simulated_data()
    else:
        # 使用模拟数据
        return load_simulated_data()

def load_simulated_data():
    """
    创建更真实的模拟 Indian Pines 数据
    """
    print("生成模拟 Indian Pines 数据...")
    
    # 真实Indian Pines的维度
    height, width, bands = 145, 145, 200
    image_shape = (height, width)
    n_pixels = height * width
    
    # 创建更真实的高光谱数据
    np.random.seed(42)
    
    # 模拟光谱特征：每个类别有不同的光谱曲线
    n_classes = 16
    class_spectra = np.zeros((n_classes, bands))
    
    # 为每个类别创建基本光谱曲线
    for c in range(1, n_classes + 1):
        # 创建有特征的光谱曲线（模拟植被、土壤、水体等）
        base_curve = np.zeros(bands)
        
        # 模拟植被在红边（波段70-80）的特征
        if c <= 9:  # 假设前9类是植被
            base_curve[70:80] = 0.8 + np.random.randn(10) * 0.1
            base_curve[30:50] = 0.3 + np.random.randn(20) * 0.05  # 可见光区域
            base_curve[100:150] = 0.5 + np.random.randn(50) * 0.1  # 近红外
        
        # 模拟土壤的光谱特征
        elif 10 <= c <= 12:
            base_curve = np.linspace(0.3, 0.6, bands) + np.random.randn(bands) * 0.05
        
        # 模拟水体的光谱特征
        elif c == 13:
            base_curve[:100] = 0.1 + np.random.randn(100) * 0.02
            base_curve[100:] = 0.05 + np.random.randn(100) * 0.01
        
        # 模拟建筑/人造物的光谱特征
        else:
            base_curve = 0.4 + np.random.randn(bands) * 0.1
        
        # 添加噪声并确保正值
        class_spectra[c-1] = np.abs(base_curve + np.random.randn(bands) * 0.02)
    
    # 创建地面真实标签 - 模拟真实的空间分布
    y = np.zeros(n_pixels, dtype=np.int8)
    
    # 创建更真实的区域分布（模拟农田地块）
    # 使用多个不同大小的矩形区域
    regions = []
    
    # 添加几个大区域
    regions.extend([
        (20, 60, 30, 80, 1),   # 区域1: 类别1（如玉米）
        (80, 120, 20, 60, 2),  # 区域2: 类别2
        (40, 90, 80, 120, 3),  # 区域3: 类别3
        (10, 40, 90, 130, 4),  # 区域4: 类别4
        (100, 130, 70, 110, 5), # 区域5: 类别5
        (60, 100, 100, 140, 6), # 区域6: 类别6
    ])
    
    # 添加一些中小区域
    regions.extend([
        (15, 35, 15, 40, 7),
        (110, 130, 10, 30, 8),
        (30, 50, 50, 70, 9),
        (90, 110, 90, 120, 10),
        (50, 70, 10, 30, 11),
        (70, 90, 60, 90, 12),
    ])
    
    # 剩余类别作为小区域或噪声
    for class_id in range(13, 17):
        # 随机位置的小区域
        y1 = np.random.randint(0, height-10)
        x1 = np.random.randint(0, width-10)
        regions.append((y1, y1+10, x1, x1+10, class_id))
    
    # 应用区域
    for y1, y2, x1, x2, class_id in regions:
        for i in range(y1, min(y2, height)):
            for j in range(x1, min(x2, width)):
                idx = i * width + j
                y[idx] = class_id
    
    # 添加一些随机噪声像素（类别16通常是小目标）
    random_indices = np.random.choice(n_pixels, size=200, replace=False)
    y[random_indices] = np.random.randint(13, 17, size=200)
    
    # 创建高光谱数据
    X = np.zeros((n_pixels, bands))
    
    # 为每个像素根据其类别分配光谱曲线
    for class_id in range(1, n_classes + 1):
        mask = y == class_id
        if np.sum(mask) > 0:
            # 使用该类别的光谱曲线作为基础
            base_spectrum = class_spectra[class_id - 1]
            
            # 为每个像素添加一些变异（模拟光谱变化）
            n_pixels_class = np.sum(mask)
            variations = np.random.randn(n_pixels_class, bands) * 0.05
            
            # 生成该类别的所有像素光谱
            X[mask] = base_spectrum + variations
            
            # 确保值为正
            X[mask] = np.abs(X[mask])
    
    # 添加全局噪声
    X += np.random.randn(n_pixels, bands) * 0.02
    X = np.abs(X)  # 确保值为正
    
    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    print(f"模拟数据生成完成: X形状={X.shape}, y形状={y.shape}, 图像形状={image_shape}")
    print(f"类别分布:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  类别 {cls}: {cnt} 像素 ({cnt/n_pixels*100:.2f}%)")
    
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

def create_rgb_image(X, image_shape, bands=[30, 20, 10]):
    """
    从高光谱数据创建RGB图像
    
    参数:
    - X: 原始数据 (n_pixels × n_bands)
    - image_shape: 图像形状 (height, width)
    - bands: [R_band, G_band, B_band] 的索引
    
    返回: RGB图像数组 (height × width × 3)
    """
    # 重塑为三维
    X_3d = X.reshape(image_shape[0], image_shape[1], -1)
    
    # 提取RGB波段
    red = X_3d[:, :, bands[0]]
    green = X_3d[:, :, bands[1]]
    blue = X_3d[:, :, bands[2]]
    
    # 分别归一化每个波段
    def normalize(band):
        band_min = band.min()
        band_max = band.max()
        if band_max - band_min > 0:
            return (band - band_min) / (band_max - band_min)
        else:
            return np.zeros_like(band)
    
    red_norm = normalize(red)
    green_norm = normalize(green)
    blue_norm = normalize(blue)
    
    # 合成RGB图像
    rgb_image = np.stack([red_norm, green_norm, blue_norm], axis=2)
    
    # 应用对比度拉伸
    rgb_image = np.clip(rgb_image * 1.2, 0, 1)  # 稍微增加对比度
    
    return rgb_image

def display_data_info(X, y, image_shape):
    """显示数据集信息"""
    print("\n" + "="*60)
    print("数据集信息")
    print("="*60)
    print(f"数据形状: {X.shape}")
    print(f"标签形状: {y.shape}")
    print(f"图像尺寸: {image_shape[0]} × {image_shape[1]} 像素")
    print(f"波段数量: {X.shape[1]}")
    print(f"总像素数: {X.shape[0]}")
    
    # 类别统计
    unique_labels = np.unique(y)
    print(f"类别数量: {len(unique_labels)}")
    print("类别分布:")
    for label in unique_labels:
        count = np.sum(y == label)
        percentage = count / len(y) * 100
        print(f"  类别 {label}: {count:5d} 像素 ({percentage:6.2f}%)")
    
    # 数据统计
    print(f"\n数据统计:")
    print(f"  最小值: {X.min():.4f}")
    print(f"  最大值: {X.max():.4f}")
    print(f"  均值: {X.mean():.4f}")
    print(f"  标准差: {X.std():.4f}")
    print("="*60 + "\n")