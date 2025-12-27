import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端，避免Tkinter线程问题
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify, send_file, url_for
import os
import numpy as np
from io import BytesIO
import base64
import json
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
try:
    from utils.data_loader import load_indian_pines_data
    from utils.classifiers import (
        apply_kmeans, apply_isodata,
        train_evaluate_svm, train_evaluate_rf
    )
except ImportError as e:
    print(f"导入模块错误: {e}")
    print("请确保utils目录中包含data_loader.py和classifiers.py文件")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/images'

# 创建必要的目录
for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULT_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# 存储当前数据和结果
current_data = {'X': None, 'y': None, 'image_shape': None, 'loaded': False}
classification_result = {'image_path': None, 'report': None, 'model': None}

@app.route('/')
def index():
    """渲染主页面"""
    return render_template('index.html')

@app.route('/load_dataset', methods=['POST'])
def load_dataset():
    """加载 Indian Pines 数据集"""
    try:
        X, y, image_shape = load_indian_pines_data()
        
        current_data['X'] = X
        current_data['y'] = y
        current_data['image_shape'] = image_shape
        current_data['loaded'] = True
        
        # 创建并保存原始数据的预览图
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # 重塑标签为图像形状
        if len(y.shape) == 1:
            y_image = y.reshape(image_shape[0], image_shape[1])
        else:
            y_image = y
        
        im = ax.imshow(y_image, cmap='jet')
        ax.set_title('Indian Pines Ground Truth', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Class Label')
        plt.tight_layout()
        
        # 保存图像到内存
        img_buf = BytesIO()
        plt.savefig(img_buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)  # 明确关闭图形，释放内存
        img_buf.seek(0)
        img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
        
        # 统计信息
        n_classes = len(np.unique(y))
        n_pixels = X.shape[0]
        n_bands = X.shape[1]
        
        return jsonify({
            'success': True,
            'message': f'数据集加载成功！形状: {X.shape}, 类别数: {n_classes}',
            'preview_image': f'data:image/png;base64,{img_base64}',
            'stats': {
                'pixels': n_pixels,
                'bands': n_bands,
                'classes': n_classes,
                'shape': f'{image_shape[0]}×{image_shape[1]}'
            }
        })
    except Exception as e:
        print(f"加载数据集时出错: {str(e)}")
        return jsonify({'success': False, 'message': f'加载数据集时出错: {str(e)}'})

@app.route('/classify', methods=['POST'])
def classify():
    """执行分类"""
    try:
        # 检查数据是否已加载
        if not current_data['loaded']:
            return jsonify({'success': False, 'message': '请先加载数据集'})
        
        # 获取前端参数
        classification_type = request.form.get('classification_type')
        algorithm = request.form.get('algorithm')
        
        X = current_data['X']
        y = current_data['y']
        image_shape = current_data['image_shape']
        
        report = ""
        result_map = None
        
        # 根据选择的分类类型和算法调用相应函数
        if classification_type == 'unsupervised':
            if algorithm == 'kmeans':
                n_clusters = int(request.form.get('n_clusters', 10))
                # 接收3个返回值：标签、报告、中心点
                result_labels, report, centers = apply_kmeans(X, n_clusters=n_clusters)
                result_name = f"kmeans_{n_clusters}clusters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                
                # 重塑为图像形状
                result_map = result_labels.reshape(image_shape[0], image_shape[1])
                
            elif algorithm == 'isodata':
                n_clusters = int(request.form.get('n_clusters', 10))
                min_samples = int(request.form.get('min_samples', 10))
                merge_threshold = float(request.form.get('merge_threshold', 2.0))
                split_threshold = float(request.form.get('split_threshold', 1.5))
                
                # 接收3个返回值：标签、报告、中心点
                result_labels, report, centers = apply_isodata(
                    X, 
                    n_clusters=n_clusters,
                    min_samples=min_samples,
                    merge_threshold=merge_threshold,
                    split_threshold=split_threshold,
                    min_clusters=5,
                    max_clusters=15
                )
                result_name = f"isodata_{n_clusters}clusters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                
                # 重塑为图像形状
                result_map = result_labels.reshape(image_shape[0], image_shape[1])
            else:
                return jsonify({'success': False, 'message': '请选择有效的非监督分类算法'})
        
        elif classification_type == 'supervised':
            test_size = float(request.form.get('test_size', 0.3))
            random_state = int(request.form.get('random_state', 42))
            
            
            if algorithm == 'svm':
                result_map, report, model= train_evaluate_svm(
                    X, y, image_shape, 
                    test_size=test_size, 
                    random_state=random_state
                )
                result_name = f"svm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                
            elif algorithm == 'rf':
                n_estimators = int(request.form.get('n_estimators', 100))
                result_map, report, model= train_evaluate_rf(
                    X, y, image_shape, 
                    test_size=test_size, 
                    random_state=random_state,
                    n_estimators=n_estimators
                )
                result_name = f"rf_{n_estimators}trees_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                
                # 保存模型到会话
                if model is not None:
                    classification_result['model'] = model
            else:
                return jsonify({'success': False, 'message': '请选择有效的监督分类算法'})
        else:
            return jsonify({'success': False, 'message': '请选择有效的分类类型'})
        
        # 可视化分类结果
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_name)
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(result_map, cmap='jet')
        
        # 设置标题
        algorithm_names = {
            'kmeans': 'K-Means',
            'isodata': 'ISODATA',
            'svm': 'SVM',
            'rf': 'Random Forest'
        }
        algo_name = algorithm_names.get(algorithm, algorithm.upper())
        title = f'{algo_name} Classification Result'
        if classification_type == 'unsupervised':
            title += f' (Clusters: {request.form.get("n_clusters", 10)})'
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Class Label')
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(result_path, dpi=120, bbox_inches='tight')
        plt.close(fig)  # 明确关闭图形
        
        # 存储结果
        classification_result['image_path'] = result_path
        classification_result['report'] = report
        
        # 将图像转换为base64用于即时预览
        with open(result_path, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        response_data = {
            'success': True,
            'message': '分类完成！',
            'result_image': f'data:image/png;base64,{img_base64}',
            'report': report,
            'result_path': f'/static/images/{result_name}'
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"分类过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'分类过程中出错: {str(e)}'})


@app.route('/download_result')
def download_result():
    """下载分类结果图像"""
    try:
        if classification_result['image_path'] and os.path.exists(classification_result['image_path']):
            return send_file(
                classification_result['image_path'], 
                as_attachment=True,
                download_name=os.path.basename(classification_result['image_path'])
            )
        return jsonify({'success': False, 'message': '结果文件不存在'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'下载文件时出错: {str(e)}'})

@app.route('/get_system_status', methods=['GET'])
def get_system_status():
    """获取系统状态"""
    status = {
        'data_loaded': current_data['loaded'],
        'has_result': classification_result['image_path'] is not None,
        'has_model': classification_result['model'] is not None,
        'result_folder': app.config['RESULT_FOLDER']
    }
    
    if current_data['loaded']:
        status['data_shape'] = current_data['X'].shape if current_data['X'] is not None else None
        status['image_shape'] = current_data['image_shape']
    
    return jsonify({
        'success': True,
        'status': status
    })

if __name__ == '__main__':
    print("=" * 60)
    print("遥感图像分类系统")
    print("=" * 60)
    print(f"访问地址: http://127.0.0.1:5000")
    print(f"静态文件目录: {app.config['RESULT_FOLDER']}")
    print("=" * 60)
    
    # 设置环境变量避免Tkinter问题
    os.environ['MPLBACKEND'] = 'Agg'
    
    app.run(debug=True, port=5000, threaded=True)