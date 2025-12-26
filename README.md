#  遥感图像分类系统

## 项目简介

这是一个基于Web的遥感图像分类系统，专门用于处理Indian Pines高光谱遥感数据集。系统提供了监督和非监督两种分类模式，支持K-means、ISODATA、SVM和随机森林四种经典分类算法。

## 数据集信息

### Indian Pines数据集
- **来源**：1992年6月在印第安纳州西北部获取
- **传感器**：AVIRIS（机载可见/红外成像光谱仪）
- **数据格式**：145×145像素，200个光谱波段
- **空间分辨率**：20米/像素
- **光谱范围**：400-2500纳米
- **地面真实类别**：16种土地覆盖类型

### 16个土地覆盖类别
1. Alfalfa (苜蓿)
2. Corn-notill (免耕玉米)
3. Corn-mintill (少耕玉米)
4. Corn (玉米)
5. Grass-pasture (草场)
6. Grass-trees (树木草地)
7. Grass-pasture-mowed (修剪草场)
8. Hay-windrowed (干草堆)
9. Oats (燕麦)
10. Soybean-notill (免耕大豆)
11. Soybean-mintill (少耕大豆)
12. Soybean-clean (净作大豆)
13. Wheat (小麦)
14. Woods (林地)
15. Buildings-Grass-Trees-Drives (建筑-草地-树木-道路)
16. Stone-Steel-Towers (石头-钢塔)

### 数据文件结构
```
data/
└── Indian_pines/
    ├── Indian_pines_corrected.mat    # 高光谱图像数据（200个波段）
    ├── Indian_pines_gt.mat           # 地面真实标签数据
    └── README.txt                    # 数据集说明
```

### 数据维度
- **原始高光谱数据**：145×145×200（高度×宽度×波段数）
- **地面真实标签**：145×145（每个像素的类别标签）
- **总像素数**：21,025个像素
- **特征维度**：每个像素200个光谱特征

## 程序结构

### 项目文件组织
```
remote_sensing_classifier/
├── app.py                    # Flask主应用程序
├── requirements.txt          # Python依赖包列表
├── README.md                # 项目说明文档
│
├── templates/               # HTML模板
│   └── index.html          # 主界面
│
├── static/                  # 静态资源
│   ├── css/                # CSS样式文件
│   └── images/             # 分类结果图像存储
│
└── utils/                   # 工具模块
    ├── data_loader.py      # 数据加载和预处理
    └── classifiers.py      # 分类器实现
```

## 运行环境要求
- **Python版本**：3.8或更高
- **操作系统**：Windows/Linux/macOS
- **内存要求**：至少4GB RAM
- **浏览器**：现代浏览器（Chrome 90+，Firefox 88+，Edge 90+）

## 依赖包
```
Flask>=2.3.0           # Web框架
numpy>=1.24.0          # 数值计算
scikit-learn>=1.3.0    # 机器学习算法
matplotlib>=3.7.0      # 数据可视化
scipy>=1.11.0          # 科学计算
Pillow>=10.0.0         # 图像处理
```

## Run

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
pip install -r requirements.txt
python app.py
```