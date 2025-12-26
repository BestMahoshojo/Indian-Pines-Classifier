import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
import os
from datetime import datetime
from scipy.spatial.distance import cdist

# ========== 非监督分类方法 ==========

def apply_kmeans(X, n_clusters=10, random_state=42, max_iter=300):
    """应用K-means聚类"""
    print(f"正在应用K-means聚类, 簇数: {n_clusters}")
    
    # 使用完整的K-means算法
    kmeans = KMeans(
        n_clusters=n_clusters, 
        random_state=random_state, 
        n_init=10,
        max_iter=max_iter,
        init='k-means++'
    )
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    
    # 计算簇内距离
    distances = np.zeros(len(X))
    for i, (point, label) in enumerate(zip(X, labels)):
        distances[i] = np.linalg.norm(point - centers[label])
    
    # 生成详细报告
    report = f"===== K-means 聚类结果 =====\n"
    report += f"簇数: {n_clusters}\n"
    report += f"数据点数量: {len(labels)}\n"
    report += f"最大迭代次数: {max_iter}\n"
    report += f"初始化方法: k-means++\n"
    report += f"收敛所需迭代次数: {kmeans.n_iter_}\n\n"
    
    # 统计每个簇的信息
    unique, counts = np.unique(labels, return_counts=True)
    report += f"聚类分布:\n"
    
    for cluster, count in zip(unique, counts):
        # 计算簇内平均距离
        cluster_indices = np.where(labels == cluster)[0]
        cluster_distances = distances[cluster_indices]
        avg_distance = np.mean(cluster_distances)
        
        report += f"簇 {cluster}: {count} 个点 ({count/len(labels)*100:.1f}%) "
        report += f"平均距离: {avg_distance:.4f}\n"
    
    # 计算总体指标
    inertia = kmeans.inertia_
    avg_cluster_distance = np.mean(distances)
    
    report += f"\n总体指标:\n"
    report += f"惯性 (inertia): {inertia:.4f}\n"
    report += f"平均簇内距离: {avg_cluster_distance:.4f}\n"
    report += f"轮廓系数估算: {1/(1+inertia/len(X)):.4f}\n"
    
    return labels, report, centers

def apply_isodata(X, n_clusters=10, max_iter=100, min_samples=10, 
                  merge_threshold=2.0, split_threshold=1.5, 
                  min_clusters=5, max_clusters=15):
    """
    应用ISODATA聚类算法 (Iterative Self-Organizing Data Analysis Technique)
    
    参数:
    - X: 输入数据
    - n_clusters: 初始簇数
    - max_iter: 最大迭代次数
    - min_samples: 簇的最小样本数
    - merge_threshold: 合并阈值 (簇中心距离小于此值则合并)
    - split_threshold: 分裂阈值 (标准差大于此值则分裂)
    - min_clusters: 最小簇数
    - max_clusters: 最大簇数
    """
    print(f"正在应用ISODATA聚类, 初始簇数: {n_clusters}")
    print(f"参数: min_samples={min_samples}, merge_threshold={merge_threshold}, split_threshold={split_threshold}")
    
    # 步骤1: 初始K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    
    n_samples, n_features = X.shape
    current_clusters = n_clusters
    
    report = f"===== ISODATA 聚类结果 =====\n"
    report += f"初始簇数: {n_clusters}\n"
    report += f"数据点数量: {n_samples}\n"
    report += f"特征数量: {n_features}\n"
    report += f"最大迭代次数: {max_iter}\n"
    report += f"最小样本数: {min_samples}\n"
    report += f"合并阈值: {merge_threshold}\n"
    report += f"分裂阈值: {split_threshold}\n"
    report += f"最小簇数: {min_clusters}\n"
    report += f"最大簇数: {max_clusters}\n\n"
    
    # 迭代优化
    for iteration in range(max_iter):
        print(f"ISODATA 迭代 {iteration + 1}/{max_iter}, 当前簇数: {current_clusters}")
        
        # 步骤2: 计算每个簇的统计信息
        unique_labels = np.unique(labels)
        current_clusters = len(unique_labels)
        
        # 存储每个簇的信息
        cluster_info = []
        for label in unique_labels:
            mask = labels == label
            cluster_points = X[mask]
            cluster_size = len(cluster_points)
            
            if cluster_size == 0:
                continue
                
            # 计算簇中心和标准差
            cluster_center = centers[label]
            if cluster_size > 1:
                cluster_std = np.std(cluster_points, axis=0).mean()  # 平均标准差
            else:
                cluster_std = 0
                
            cluster_info.append({
                'label': label,
                'size': cluster_size,
                'center': cluster_center,
                'std': cluster_std,
                'points': cluster_points,
                'indices': np.where(mask)[0]
            })
        
        # 步骤3: 删除小簇 (重新分配给最近的簇)
        clusters_to_remove = []
        for info in cluster_info:
            if info['size'] < min_samples and current_clusters > min_clusters:
                clusters_to_remove.append(info['label'])
                
        if clusters_to_remove:
            report += f"迭代 {iteration + 1}: 发现小簇 {clusters_to_remove}\n"
            
        for label_to_remove in clusters_to_remove:
            # 找到这个簇的点
            indices_to_reassign = np.where(labels == label_to_remove)[0]
            
            if len(indices_to_reassign) == 0:
                continue
                
            # 为每个点找到最近的簇 (排除要删除的簇)
            valid_labels = [l for l in unique_labels if l != label_to_remove]
            if len(valid_labels) == 0:
                continue
                
            # 计算到所有其他簇中心的距离
            points_to_reassign = X[indices_to_reassign]
            valid_centers = np.array([centers[l] for l in valid_labels])
            
            # 为每个点分配新的标签
            for i, idx in enumerate(indices_to_reassign):
                distances = np.linalg.norm(points_to_reassign[i] - valid_centers, axis=1)
                nearest_idx = np.argmin(distances)
                labels[idx] = valid_labels[nearest_idx]
            
            current_clusters -= 1
            report += f"  删除簇 {label_to_remove} ({len(indices_to_reassign)}个点重新分配)\n"
        
        # 更新簇信息
        unique_labels = np.unique(labels)
        current_clusters = len(unique_labels)
        
        # 重新计算簇中心
        for label in unique_labels:
            mask = labels == label
            if np.sum(mask) > 0:
                centers[label] = X[mask].mean(axis=0)
        
        # 步骤4: 合并相近的簇
        if current_clusters > min_clusters:
            # 计算所有簇中心之间的距离
            cluster_centers = np.array([centers[l] for l in unique_labels])
            distances = cdist(cluster_centers, cluster_centers)
            np.fill_diagonal(distances, np.inf)  # 将对角线设为无穷大
            
            # 寻找需要合并的簇对
            merge_pairs = []
            for i, label_i in enumerate(unique_labels):
                for j, label_j in enumerate(unique_labels[i+1:], i+1):
                    if distances[i, j] < merge_threshold:
                        merge_pairs.append((label_i, label_j, distances[i, j]))
            
            # 合并簇
            if merge_pairs:
                report += f"迭代 {iteration + 1}: 发现可合并的簇对 {len(merge_pairs)}\n"
                
            for label_i, label_j, distance in merge_pairs:
                if current_clusters <= min_clusters:
                    break
                    
                # 将两个簇合并 (保留较小的标签)
                labels[labels == label_j] = label_i
                current_clusters -= 1
                report += f"  合并簇 {label_j} 到 {label_i} (距离: {distance:.4f})\n"
        
        # 步骤5: 分裂标准差过大的簇
        if current_clusters < max_clusters:
            clusters_to_split = []
            
            # 重新计算簇信息
            cluster_info = []
            for label in unique_labels:
                mask = labels == label
                cluster_points = X[mask]
                cluster_size = len(cluster_points)
                
                if cluster_size < 2:  # 需要至少2个点才能计算标准差
                    continue
                    
                # 计算簇中心和标准差
                cluster_center = centers[label]
                cluster_std = np.std(cluster_points, axis=0).mean()
                
                cluster_info.append({
                    'label': label,
                    'size': cluster_size,
                    'center': cluster_center,
                    'std': cluster_std,
                    'points': cluster_points
                })
                
                # 检查是否需要分裂
                if cluster_std > split_threshold and cluster_size > 2 * min_samples:
                    clusters_to_split.append(label)
            
            # 分裂簇
            if clusters_to_split:
                report += f"迭代 {iteration + 1}: 发现可分裂的簇 {clusters_to_split}\n"
                
            for label in clusters_to_split:
                if current_clusters >= max_clusters:
                    break
                    
                mask = labels == label
                cluster_points = X[mask]
                
                if len(cluster_points) < 2:
                    continue
                
                # 在簇内运行K-means (2个簇)
                try:
                    sub_kmeans = KMeans(n_clusters=2, random_state=42, n_init=5)
                    sub_labels = sub_kmeans.fit_predict(cluster_points)
                    
                    # 创建新标签
                    new_label = max(unique_labels) + 1
                    
                    # 分配新标签
                    cluster_indices = np.where(mask)[0]
                    for idx, sub_label in zip(cluster_indices, sub_labels):
                        if sub_label == 1:
                            labels[idx] = new_label
                    
                    # 更新簇中心
                    if np.sum(labels == label) > 0:
                        centers[label] = X[labels == label].mean(axis=0)
                    if np.sum(labels == new_label) > 0:
                        centers = np.vstack([centers, X[labels == new_label].mean(axis=0)])
                    
                    current_clusters += 1
                    report += f"  分裂簇 {label} (标准差: {cluster_std:.4f}) -> 创建新簇 {new_label}\n"
                    
                except Exception as e:
                    report += f"  分裂簇 {label} 失败: {str(e)}\n"
        
        # 步骤6: 检查收敛 (簇数量稳定)
        if iteration > 0 and current_clusters == prev_clusters:
            # 检查簇标签是否变化不大
            label_changes = np.sum(labels != prev_labels)
            if label_changes < n_samples * 0.01:  # 少于1%的点发生变化
                report += f"\n收敛于迭代 {iteration + 1}: 簇标签变化小于1%\n"
                break
        
        prev_labels = labels.copy()
        prev_clusters = current_clusters
    
    # 最终统计
    unique_labels, counts = np.unique(labels, return_counts=True)
    final_clusters = len(unique_labels)
    
    report += f"\n最终结果:\n"
    report += f"迭代次数: {min(iteration + 1, max_iter)}\n"
    report += f"最终簇数: {final_clusters}\n"
    report += f"数据点数量: {n_samples}\n\n"
    
    # 计算每个簇的详细信息
    total_inertia = 0
    for label in unique_labels:
        mask = labels == label
        cluster_points = X[mask]
        cluster_size = len(cluster_points)
        
        if cluster_size == 0:
            continue
            
        # 计算簇中心
        cluster_center = cluster_points.mean(axis=0)
        
        # 计算簇内距离平方和
        distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
        cluster_inertia = np.sum(distances ** 2)
        total_inertia += cluster_inertia
        
        # 计算平均距离和标准差
        avg_distance = np.mean(distances)
        cluster_std = np.std(cluster_points, axis=0).mean() if cluster_size > 1 else 0
        
        report += f"簇 {label}: {cluster_size} 个点 ({cluster_size/n_samples*100:.1f}%)\n"
        report += f"  平均距离: {avg_distance:.4f}, 标准差: {cluster_std:.4f}, 惯性: {cluster_inertia:.4f}\n"
    
    report += f"\n总体惯性 (inertia): {total_inertia:.4f}\n"
    report += f"平均轮廓系数估算: {1/(1+total_inertia/n_samples):.4f}\n"
    
    print(f"ISODATA 完成: 初始 {n_clusters} 个簇 -> 最终 {final_clusters} 个簇")
    
    return labels, report, centers

# ========== 监督分类方法 ==========

def train_evaluate_svm(X, y, image_shape, test_size=0.3, random_state=42, 
                       model_name=None):
    """训练并评估SVM分类器"""
    from sklearn.model_selection import train_test_split
    
    print("正在训练SVM分类器...")
    
    # 准备数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 训练SVM
    clf = svm.SVC(kernel='linear', random_state=random_state, probability=True)
    clf.fit(X_train, y_train)
    
    # 预测
    y_pred = clf.predict(X_test)
    
    # 在整个图像上生成分类结果
    full_pred = clf.predict(X)
    result_map = full_pred.reshape(image_shape[0], image_shape[1])
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    report_str = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # 生成详细报告
    report = f"===== SVM 分类器评估结果 =====\n"
    report += f"测试集准确率: {accuracy:.4f}\n"
    report += f"训练样本数: {len(X_train)}\n"
    report += f"测试样本数: {len(X_test)}\n"
    report += f"类别数: {len(np.unique(y))}\n\n"
    report += f"分类报告:\n{report_str}\n"
    
    # 混淆矩阵摘要
    report += f"混淆矩阵摘要:\n"
    report += f"  对角线(正确分类): {np.trace(cm)} 个样本\n"
    report += f"  总样本数: {np.sum(cm)}\n"
    
    return result_map, report, clf

def train_evaluate_rf(X, y, image_shape, test_size=0.3, random_state=42, 
                      n_estimators=100,model_name=None):
    """训练并评估随机森林分类器"""
    from sklearn.model_selection import train_test_split
    
    print(f"正在训练随机森林分类器 (树数量: {n_estimators})...")
    
    # 准备数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 训练随机森林
    clf = RandomForestClassifier(
        n_estimators=n_estimators, 
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    # 预测
    y_pred = clf.predict(X_test)
    
    # 在整个图像上生成分类结果
    full_pred = clf.predict(X)
    result_map = full_pred.reshape(image_shape[0], image_shape[1])
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    report_str = classification_report(y_test, y_pred)
    
    # 生成详细报告
    report = f"===== 随机森林分类器评估结果 =====\n"
    report += f"测试集准确率: {accuracy:.4f}\n"
    report += f"树的数量: {n_estimators}\n"
    report += f"训练样本数: {len(X_train)}\n"
    report += f"测试样本数: {len(X_test)}\n"
    report += f"类别数: {len(np.unique(y))}\n\n"
    report += f"分类报告:\n{report_str}\n"
    
    # 特征重要性
    report += f"特征重要性 (前10个波段):\n"
    importances = clf.feature_importances_
    top_10_idx = np.argsort(importances)[-10:][::-1]
    for i, idx in enumerate(top_10_idx):
        report += f"  波段 {idx+1}: {importances[idx]:.4f}\n"
    
    return result_map, report, clf

# ========== 可视化辅助函数 ==========

def visualize_clusters(X, labels, centers, title="聚类结果"):
    """可视化聚类结果（降维显示）"""
    from sklearn.decomposition import PCA
    
    # 使用PCA降维到2D以便可视化
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    centers_2d = pca.transform(centers)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制数据点
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab20', 
                         alpha=0.6, s=10)
    
    # 绘制簇中心
    ax.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', 
               marker='X', s=200, edgecolors='black', linewidth=2)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    
    plt.colorbar(scatter, ax=ax, label='Cluster Labels')
    plt.tight_layout()
    
    return fig