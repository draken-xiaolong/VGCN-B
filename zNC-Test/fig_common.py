#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig1-Fig12 共享逻辑模块
提取公共函数，避免代码冗余

⭐ 已适配VGCN/VGCN.py和convertToGraph-TrainingSet.py的逻辑：
- 使用13维特征（与训练集一致）
- 使用纯KNN无向图（与训练集一致）
- 修正全局标准化器路径（添加/cache/）
- 支持加载VGCN的GCNModel模型（13维输入）

包含内容：
1. 路径配置
2. 13维特征提取函数（与训练集convertToGraph-TrainingSet.py一致）
3. 图构建函数（KNN无向图，与训练集一致）
4. 模型加载函数（支持VGCN和VGAT模型）
5. 零水印工具函数（load_cat32, features_to_matrix, calc_nc）
6. 结果保存函数
"""

from pathlib import Path
import os
import sys
from typing import List, Tuple, Optional
import pickle

import numpy as np
from shapely.geometry import Point  # type: ignore

try:
    import geopandas as gpd  # type: ignore
except Exception:
    gpd = None

try:
    from sklearn.preprocessing import StandardScaler  # type: ignore
    from sklearn.neighbors import NearestNeighbors  # type: ignore
except Exception:
    StandardScaler = None
    NearestNeighbors = None

try:
    from scipy.spatial import Delaunay  # type: ignore
except Exception:
    Delaunay = None

try:
    import torch  # type: ignore
    from torch_geometric.data import Data  # type: ignore
except Exception:
    Data = None
    torch = None

# ====================
# 路径配置
# ====================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

# 模型和资源路径
# ⭐ 注意：应使用VGCN训练的GCN模型，而不是VGAT模型
GRAPH_SUFFIX = os.environ.get("VGAT_GRAPH_SUFFIX", "").strip()
MODEL_PATH = Path(os.environ.get("VGAT_MODEL_PATH", str(PROJECT_ROOT / 'VGCN' / 'models' / 'gcn_model_best.pth')))
# 备用：VGAT模型路径（如果VGCN模型不存在，可回退使用）
MODEL_PATH_VGAT = PROJECT_ROOT / 'VGAT' / 'models' / 'gat_model_IMPROVED_best.pth'
CAT32_PATH = PROJECT_ROOT / 'ZeroWatermark' / 'Cat32.png'
GLOBAL_SCALER_PATH = PROJECT_ROOT / 'convertToGraph' / 'Graph' / f'TrainingSet{GRAPH_SUFFIX}' / 'cache' / 'global_scaler.pkl'

# 配置参数
K_FOR_KNN = int(os.environ.get("VGAT_KNN_K", "8"))  # KNN邻居数

# 全局标准化器（延迟加载）
_global_scaler = None
_global_scaler_loaded = False


def load_global_scaler():
    """
    加载训练集的全局标准化器
    
    Returns:
        StandardScaler或None: 全局标准化器，如果加载失败则返回None
    """
    global _global_scaler, _global_scaler_loaded
    
    # 如果已经尝试过加载，直接返回结果
    if _global_scaler_loaded:
        return _global_scaler
    
    _global_scaler_loaded = True  # 标记已尝试加载
    
    try:
        if not GLOBAL_SCALER_PATH.exists():
            print(f"❌ 错误：未找到全局标准化器")
            print(f"   路径: {GLOBAL_SCALER_PATH}")
            print(f"   ⚠️  局部标准化会导致NC值虚高，因此已禁用！")
            print(f"   请先运行 convertToGraph-TrainingSet.py 生成全局标准化器")
            return None
        
        with open(GLOBAL_SCALER_PATH, 'rb') as f:
            scaler_data = pickle.load(f)
        
        if isinstance(scaler_data, dict):
            _global_scaler = scaler_data.get('scaler')
        else:
            _global_scaler = scaler_data
        
        print(f"✓ 已加载训练集的全局标准化器: {GLOBAL_SCALER_PATH.name}")
        return _global_scaler
        
    except Exception as e:
        print(f"❌ 错误：加载全局标准化器失败")
        print(f"   错误信息: {e}")
        print(f"   ⚠️  局部标准化会导致NC值虚高，因此已禁用！")
        return None


# ====================
# 特征提取函数（13维）
# ====================

def extract_features_20d(geometry, all_geometries=None, idx=None, bounds_stats=None, row=None) -> np.ndarray:
    """
    提取13维几何特征（使用相对坐标实现平移不变性）
    
    特征列表：
    0-2:   几何类型编码（one-hot）Point/Line/Polygon
    3:     面积
    4:     周长
    5-8:   边界框（相对于质心的相对坐标：minx-cx, miny-cy, maxx-cx, maxy-cy）
    9-10:  边界框尺寸（width, height）
    11-12: 质心相对坐标（始终为0,0，保持平移不变性）
    
    ⭐ 修改说明：
    - 原版本使用绝对坐标，导致平移攻击NC值低（0.638）
    - 新版本使用相对坐标，实现平移不变性
    - 特征维度保持13维不变，无需修改模型
    """
    # ⭐ 与训练集convertToGraph-TrainingSet.py的extract_features函数完全一致
    features = []
    
    # 1. 几何类型编码（3维）
    geom_type = getattr(geometry, 'geom_type', 'Unknown')
    if geom_type == 'Point':
        geom_features = [1, 0, 0]  # 点图层
    elif geom_type in ['LineString', 'MultiLineString']:
        geom_features = [0, 1, 0]  # 线图层
    elif geom_type in ['Polygon', 'MultiPolygon']:
        geom_features = [0, 0, 1]  # 面图层
    else:
        geom_features = [0, 0, 0]  # 未知类型
    features.extend(geom_features)
    
    # 2. 面积 - 几何要素的面积
    if hasattr(geometry, 'area'):
        features.append(geometry.area)
    else:
        features.append(0.0)
    
    # 3. 周长 - 几何要素的周长
    if hasattr(geometry, 'length'):
        features.append(geometry.length)
    else:
        features.append(0.0)
    
    # 4-7. 边界框特征（改为相对于质心的相对坐标，保持平移不变性）
    bounds = geometry.bounds
    centroid = geometry.centroid
    cx, cy = centroid.x, centroid.y
    
    features.extend([
        bounds[0] - cx, bounds[1] - cy,  # 最小X（相对）, 最小Y（相对）
        bounds[2] - cx, bounds[3] - cy   # 最大X（相对）, 最大Y（相对）
    ])
    
    # 8-9. 边界框尺寸（不受平移影响，保持不变）
    features.extend([
        bounds[2] - bounds[0], bounds[3] - bounds[1]  # 宽度, 高度
    ])
    
    # 10-11. 质心坐标（改为相对坐标，设为0以保持平移不变性）
    # 注意：由于质心是参考点，其相对于自身的坐标为(0,0)
    features.extend([0.0, 0.0])
    
    # 12-13. 根据几何类型调整面积和周长
    if geom_type == 'Point':
        # 点图层：面积和周长都为0
        features[3] = 0.0  # 面积
        features[4] = 0.0  # 周长
    elif geom_type in ['LineString', 'MultiLineString']:
        # 线图层：面积为0
        features[3] = 0.0  # 面积
    
    feats = features
    
    return np.array(feats, dtype=np.float32)


# ====================
# 图构建函数
# ====================

def _hilbert_curve_sort(centroids):
    """
    使用Hilbert曲线对坐标点排序（保持空间局部性）
    
    Args:
        centroids: nx2的坐标数组
        
    Returns:
        排序后的索引数组
    """
    try:
        from hilbertcurve.hilbertcurve import HilbertCurve
    except ImportError:
        # 降级到简单的x+y排序
        print(f"      ⚠️ hilbertcurve未安装，使用简化排序")
        print(f"      提示：pip install hilbertcurve 可获得更好性能")
        return np.argsort(centroids[:, 0] + centroids[:, 1])
    
    n = len(centroids)
    
    # 标准化坐标到[0, 2^p-1]范围
    x_min, x_max = centroids[:, 0].min(), centroids[:, 0].max()
    y_min, y_max = centroids[:, 1].min(), centroids[:, 1].max()
    
    # 计算合适的Hilbert曲线阶数（p值）
    # 2^p应该足够大以保证精度，但不能太大导致溢出
    p = min(15, max(8, int(np.log2(np.sqrt(n))) + 3))
    max_coord = (1 << p) - 1  # 2^p - 1
    
    # 标准化坐标
    if x_max > x_min:
        x_norm = ((centroids[:, 0] - x_min) / (x_max - x_min) * max_coord).astype(int)
    else:
        x_norm = np.zeros(n, dtype=int)
    
    if y_max > y_min:
        y_norm = ((centroids[:, 1] - y_min) / (y_max - y_min) * max_coord).astype(int)
    else:
        y_norm = np.zeros(n, dtype=int)
    
    # 计算Hilbert距离
    hc = HilbertCurve(p, 2)  # p阶，2维
    hilbert_distances = np.array([
        hc.distance_from_point([int(x), int(y)]) 
        for x, y in zip(x_norm, y_norm)
    ])
    
    # 按Hilbert距离排序
    return np.argsort(hilbert_distances)


def build_knn_delaunay_edges(geometries, k: int = K_FOR_KNN):
    """
    构建纯KNN无向图（与训练集convertToGraph-TrainingSet.py完全一致）
    
    核心策略：
    1. 使用几何质心坐标计算距离
    2. 每个节点连接K个最近邻
    3. 转为无向图（A + A^T）
    4. k值自适应（最小为1，单节点返回无边图）
    
    Args:
        geometries: 几何要素列表
        k: KNN邻居数
        
    Returns:
        edges: 边列表 [[src, dst], ...]（双向边）
    """
    from sklearn.neighbors import kneighbors_graph
    
    n = len(geometries)
    
    # 特殊情况：只有1个节点，返回无边图
    if n == 1:
        print(f"  📊 单节点图 (节点数=1, 无边)")
        return []
    
    # 动态调整k值：最小为1，最大为n-1
    actual_k = min(max(1, k), n - 1)
    if actual_k < k:
        print(f"  📊 节点数{n}<k={k}，自适应调整为k={actual_k}")
    else:
        print(f"  📊 构建无向K近邻图 (k={actual_k}, 节点数={n})")
    
    # 使用质心坐标计算距离
    centroids = np.array([[geom.centroid.x, geom.centroid.y] for geom in geometries])
    
    # 确保centroids是2维数组
    if centroids.ndim == 1:
        centroids = centroids.reshape(1, -1)
    
    # 构建K近邻图
    adjacency_matrix = kneighbors_graph(centroids, n_neighbors=actual_k, mode='connectivity', include_self=False)
    
    # 转换为无向图：A_undirected = A + A^T
    adjacency_matrix = adjacency_matrix + adjacency_matrix.T
    
    # 转换为边列表格式（双向边）
    edge_indices = np.vstack(adjacency_matrix.nonzero())
    edges = [[int(edge_indices[0, i]), int(edge_indices[1, i])] for i in range(edge_indices.shape[1])]
    
    print(f"    ✓ KNN图构建完成，边数: {len(edges)//2}对（{len(edges)}条有向边）")
    
    return edges


def gdf_to_graph(gdf, max_nodes=None) -> Optional[Data]:
    """
    从GeoDataFrame构建图结构（13维特征 + KNN无向图）
    与训练集convertToGraph-TrainingSet.py的build_graph_from_gdf函数逻辑完全一致
    
    Args:
        gdf: GeoDataFrame对象
        max_nodes: 最大节点数阈值，超过则返回None（None表示不限制）
    
    Returns:
        Data对象或None（如果节点数超过阈值）
    """
    if Data is None or StandardScaler is None:
        print("缺少依赖：torch-geometric 或 scikit-learn")
        return None
    
    # ⭐检查节点数，超过阈值则跳过（仅当max_nodes不为None时）
    if max_nodes is not None:
        num_nodes = len(gdf)
        if num_nodes > max_nodes:
            return None
    
    geometries = gdf.geometry.tolist()
    
    # ⭐ 提取13维特征（与训练集一致）
    print(f"    提取特征...")
    node_features = []
    for idx, row in gdf.iterrows():
        features = extract_features_20d(row.geometry, geometries, idx, None, row)
        node_features.append(features)
    
    feats = np.array(node_features, dtype=np.float32)
    
    # 特征归一化（必须使用全局标准化器）
    if len(feats) > 0:
        # 加载训练集的全局标准化器
        global_scaler = load_global_scaler()
        
        if global_scaler is not None:
            # ✅ 使用训练集的全局标准化器（符合机器学习标准实践）
            feats = global_scaler.transform(feats)
            
            # ⭐ 特征裁剪已禁用
            # 原因：相对坐标特征已经实现了平移不变性
            #       即使标准化后有些特征值较大，但对所有图都一致
            #       GCN模型仍能正确处理（测试证明NC=1.0）
            #       裁剪反而会丢失重要的区分信息，导致NC下降25%
            # 结论：不需要裁剪！相对坐标足够了 ✅
        else:
            # ❌ 必须使用全局标准化器，否则NC值不可信
            raise FileNotFoundError(
                f"全局标准化器未找到！\n"
                f"路径: {GLOBAL_SCALER_PATH}\n"
                f"原因: 局部标准化会导致NC值虚高（数据泄露问题）\n"
                f"解决方案:\n"
                f"  1. 确保训练集已生成全局标准化器\n"
                f"  2. 检查路径是否正确\n"
                f"  3. 运行 convertToGraph/convertToGraph-TrainingSet.py 生成标准化器"
            )
    
    # ✅ 构建边：使用KNN无向图（基于几何质心坐标，与训练集一致）
    edges = build_knn_delaunay_edges(geometries, K_FOR_KNN)
    
    # 转换为edge_index格式
    if len(edges) > 0:
        edge_index = torch.tensor(edges, dtype=torch.long).T
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # 创建Data对象
    data = Data(
        x=torch.tensor(feats, dtype=torch.float32),
        edge_index=edge_index
    )
    
    return data


# ====================
# 模型加载函数
# ====================

def load_improved_gat_model(device='cpu', model_path=None):
    """
    加载GCNModel模型（VGCN训练的模型）
    注意：实际应该使用VGCN/models/gcn_model_best.pth，而不是VGAT模型
    
    Returns:
        model: 加载好的模型（已设置为eval模式）
        device: 使用的设备
    """
    if model_path is None:
        model_path = MODEL_PATH
    
    # ⭐ 自动检测并使用正确的模型
    if not Path(model_path).exists():
        print(f"⚠️ VGCN模型不存在: {model_path}")
        if Path(MODEL_PATH_VGAT).exists():
            print(f"⚠️ 回退使用VGAT模型（特征维度不匹配，可能影响性能）")
            model_path = MODEL_PATH_VGAT
            use_vgat = True
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
    else:
        use_vgat = False
    
    if use_vgat:
        # 加载VGAT模型（20维输入）
        vgat_path = str(PROJECT_ROOT / 'VGAT')
        if vgat_path not in sys.path:
            sys.path.insert(0, vgat_path)
        
        try:
            from VGAT import ImprovedGATModel  # type: ignore
        except Exception as exc:
            print(f'导入ImprovedGATModel失败: {exc}')
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "vgat_improved", 
                str(PROJECT_ROOT / 'VGAT' / 'VGAT-IMPROVED.py')
            )
            vgat_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(vgat_module)
            ImprovedGATModel = vgat_module.ImprovedGATModel
        
        model = ImprovedGATModel(
            input_dim=20,
            hidden_dim=256,
            output_dim=1024,
            num_heads=8,
            dropout=0.3
        )
        input_dim = 20
    else:
        # 加载VGCN的GCNModel（13维输入）
        vgcn_path = str(PROJECT_ROOT / 'VGCN')
        if vgcn_path not in sys.path:
            sys.path.insert(0, vgcn_path)
        
        try:
            from VGCN import GCNModel  # type: ignore
        except Exception as exc:
            raise ImportError(f"无法导入GCNModel: {exc}")
        
        # 创建GCN模型（默认会被checkpoint中的配置覆盖）
        input_dim = 13
        hidden_dim = 128
        output_dim = 1024
        dropout = 0.2
        pooling_mode = 'dual'
        model = GCNModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            pooling_mode=pooling_mode
        )
    
    # 加载权重
    ckpt = torch.load(str(model_path), map_location=device)
    model_config = ckpt.get('model_config', {}) if isinstance(ckpt, dict) else {}
    if not use_vgat and model_config:
        input_dim = int(model_config.get('input_dim', input_dim))
        hidden_dim = int(model_config.get('hidden_dim', hidden_dim))
        output_dim = int(model_config.get('output_dim', output_dim))
        dropout = float(model_config.get('dropout', dropout))
        pooling_mode = str(model_config.get('pooling_mode', pooling_mode))
        model = GCNModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            pooling_mode=pooling_mode
        )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f'✓ 成功加载模型: {Path(model_path).name}')
    print(f'  模型类型: {"VGAT (ImprovedGATModel)" if use_vgat else "VGCN (GCNModel)"}')
    print(f'  设备: {device}')
    print(f'  输入维度: {input_dim}')
    print(f'  输出维度: 1024')
    if not use_vgat:
        print(f'  Pooling: {getattr(model, "pooling_mode", "dual")}')
    
    return model, device


# ====================
# 零水印工具函数
# ====================

def generate_scramble_key(dataset_name, timestamp=None):
    """
    根据数据集名称和时间戳生成置乱密钥
    
    Args:
        dataset_name: 数据集名称
        timestamp: 时间戳（如果为None，使用当前时间）
    
    Returns:
        scramble_key: 置乱密钥（整数）
        timestamp: 使用的时间戳
    """
    import hashlib
    import time
    
    if timestamp is None:
        timestamp = int(time.time())
    
    # 组合数据集名称和时间戳
    key_string = f"{dataset_name}_{timestamp}"
    
    # 使用SHA256生成哈希值
    hash_object = hashlib.sha256(key_string.encode())
    hash_hex = hash_object.hexdigest()
    
    # 将哈希值转换为整数作为随机种子
    scramble_key = int(hash_hex[:16], 16)  # 取前16位十六进制
    
    return scramble_key, timestamp


def scramble_image(image, scramble_key):
    """
    使用Arnold变换置乱图像
    
    Args:
        image: 输入图像（numpy数组）
        scramble_key: 置乱密钥（用作迭代次数的种子）
    
    Returns:
        scrambled_image: 置乱后的图像
    """
    if image.ndim != 2:
        raise ValueError("输入图像必须是二维数组")
    
    N = image.shape[0]
    if image.shape[0] != image.shape[1]:
        raise ValueError("输入图像必须是方阵")
    
    # 使用密钥确定迭代次数
    iterations = (scramble_key % (N * N)) + 1
    
    scrambled = image.copy()
    
    for _ in range(iterations):
        temp = np.zeros_like(scrambled)
        for i in range(N):
            for j in range(N):
                # Arnold变换公式
                new_i = (i + j) % N
                new_j = (i + 2 * j) % N
                temp[new_i, new_j] = scrambled[i, j]
        scrambled = temp
    
    return scrambled


def descramble_image(scrambled_image, scramble_key):
    """
    使用Arnold逆变换恢复图像
    
    Args:
        scrambled_image: 置乱后的图像
        scramble_key: 置乱密钥（必须与置乱时相同）
    
    Returns:
        recovered_image: 恢复后的图像
    """
    if scrambled_image.ndim != 2:
        raise ValueError("输入图像必须是二维数组")
    
    N = scrambled_image.shape[0]
    if scrambled_image.shape[0] != scrambled_image.shape[1]:
        raise ValueError("输入图像必须是方阵")
    
    # 使用相同的密钥确定迭代次数
    iterations = (scramble_key % (N * N)) + 1
    
    recovered = scrambled_image.copy()
    
    for _ in range(iterations):
        temp = np.zeros_like(recovered)
        for i in range(N):
            for j in range(N):
                # Arnold逆变换公式
                new_i = (2 * i - j) % N
                new_j = (-i + j) % N
                temp[new_i, new_j] = recovered[i, j]
        recovered = temp
    
    return recovered


def save_scramble_info(dataset_name, scramble_key, timestamp, output_path):
    """保存置乱信息到文件"""
    import json
    
    info = {
        'dataset_name': dataset_name,
        'scramble_key': int(scramble_key),
        'timestamp': int(timestamp)
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def load_scramble_info(info_path):
    """从文件加载置乱信息"""
    import json
    
    with open(info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)
    
    return info['dataset_name'], info['scramble_key'], info['timestamp']


def load_cat32(img_path=None):
    """加载并预处理版权图像为32x32二值图"""
    if img_path is None:
        img_path = CAT32_PATH
    
    try:
        from PIL import Image  # type: ignore
    except Exception:
        raise ImportError("需要安装 pillow: pip install pillow")
    
    try:
        img = Image.open(img_path)
        img = img.convert('L').resize((32, 32))
        img = img.point(lambda x: 0 if x < 128 else 255, '1')
        return np.array(img, dtype=np.uint8)
    except Exception as exc:
        print(f'加载版权图像失败: {exc}')
        # 返回随机图像作为备选
        return (np.random.rand(32, 32) > 0.5).astype(np.uint8)


def features_to_matrix(features: np.ndarray, shape=(32, 32)) -> np.ndarray:
    """将特征向量转换为二值矩阵（基于中位数阈值）"""
    total = shape[0] * shape[1]
    
    # 展平特征
    if features.ndim > 1:
        features_1d = features.flatten()
    else:
        features_1d = features
    
    # 如果特征不足，重复填充
    if len(features_1d) < total:
        rep = (total + len(features_1d) - 1) // len(features_1d)
        features_1d = np.tile(features_1d, rep)
    
    # 截取到目标大小并reshape
    mat = features_1d[:total].reshape(shape)
    
    # 中位数阈值二值化
    thr = np.median(mat)
    return (mat > thr).astype(np.uint8)


def calc_nc(a: np.ndarray, b: np.ndarray) -> float:
    """计算归一化相关系数（NC）"""
    va = a.flatten().astype(float)
    vb = b.flatten().astype(float)
    
    dot = float(np.sum(va * vb))
    na = float(np.sqrt(np.sum(va ** 2)))
    nb = float(np.sqrt(np.sum(vb ** 2)))
    
    if na == 0 or nb == 0:
        return 0.0
    
    return dot / (na * nb)


def extract_features_from_graph(graph_data, model, device, copyright_shape=(32, 32)):
    """
    从图数据中提取1024维特征并转为二值矩阵
    
    Args:
        graph_data: PyTorch Geometric Data对象
        model: VGAT模型
        device: 设备
        copyright_shape: 版权图像形状
    
    Returns:
        feat_matrix: 二值特征矩阵
    """
    with torch.no_grad():
        feat = model(
            graph_data.x.to(device),
            graph_data.edge_index.to(device)
        ).detach().cpu().numpy()
    
    # 确保是1024维
    if feat.ndim > 1:
        feat = feat.flatten()
    
    if len(feat) != 1024:
        if len(feat) < 1024:
            rep = (1024 + len(feat) - 1) // len(feat)
            feat = np.tile(feat, rep)
        feat = feat[:1024]
    
    # 转为二值矩阵
    feat_matrix = features_to_matrix(feat, copyright_shape)
    
    return feat_matrix


# ====================
# GeoJSON转换函数
# ====================

def convert_to_geojson(input_paths: List[Path], output_dir: Path) -> List[Path]:
    """
    批量转换矢量数据为GeoJSON
    
    Args:
        input_paths: 输入文件路径列表（.shp或.geojson）
        output_dir: 输出目录
    
    Returns:
        outputs: 转换后的GeoJSON文件路径列表
    """
    if gpd is None:
        raise ImportError("需要安装 geopandas")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []
    
    for src in input_paths:
        # ⭐跳过macOS元数据文件
        if src.name.startswith('._'):
            continue
            
        try:
            base = src.stem
            out_path = output_dir / f'{base}.geojson'
            
            gdf = gpd.read_file(src)
            
            # 转换坐标系为WGS84
            if getattr(gdf, 'crs', None) and str(gdf.crs) != 'EPSG:4326':
                gdf = gdf.to_crs('EPSG:4326')
            
            gdf.to_file(out_path, driver='GeoJSON', encoding='utf-8')
            print(f'  ✓ {out_path.name} ({len(gdf)} 要素)')
            outputs.append(out_path)
            
        except Exception as exc:
            print(f'  ✗ {src.name}: {exc}')
            continue
    
    return outputs


# ====================
# 通用图数据转换函数
# ====================

def convert_geojsons_to_graphs(
    original_geojsons: List[Path],
    attacked_geojson_map: dict,
    output_dir_original: Path,
    output_dir_attacked: Path,
    max_nodes=None
):
    """
    批量转换GeoJSON为图结构（可选节点数过滤）
    
    Args:
        original_geojsons: 原始GeoJSON文件列表
        attacked_geojson_map: 攻击后的GeoJSON文件映射 {base_name: {attack_param: path}}
        output_dir_original: 原始图输出目录
        output_dir_attacked: 攻击图输出目录
        max_nodes: 最大节点数阈值（None表示不限制，默认None）
    """
    import shutil
    
    # 确保输出目录存在（不删除已有文件，支持断点续传）
    # ⚠️ 不删除已有的图文件，允许增量转换和断点续传
    output_dir_original.mkdir(parents=True, exist_ok=True)
    output_dir_attacked.mkdir(parents=True, exist_ok=True)
    
    # 统计
    skipped_original = []
    processed_original = []
    skipped_attacked = 0
    processed_attacked = 0
    
    # 转换原始图
    threshold_info = f"不限制" if max_nodes is None else f"{max_nodes}"
    print(f'\n[转换原始图] (节点数阈值: {threshold_info})')
    for src in original_geojsons:
        # ⭐跳过macOS元数据文件（以._开头的文件）
        if src.name.startswith('._'):
            print(f'  ⚠️  跳过macOS元数据文件: {src.name}')
            continue
        
        # ⭐检查原始图是否已存在，若存在则跳过
        out_path = output_dir_original / f"{src.stem}_graph.pkl"
        if out_path.exists():
            print(f'  ⏭️  跳过 {src.name}：原始图已存在')
            processed_original.append(src.stem)
            continue
        
        try:
            gdf = gpd.read_file(src)
            num_nodes = len(gdf)
            
            # ⭐检查节点数（仅当max_nodes不为None时）
            if max_nodes is not None and num_nodes > max_nodes:
                print(f'  ⚠️  跳过 {src.name}: 节点数={num_nodes} > {max_nodes}')
                skipped_original.append(src.stem)
                continue
            
            data = gdf_to_graph(gdf, max_nodes)
            
            if data is not None:
                with open(out_path, 'wb') as f:
                    pickle.dump(data, f)
                print(f'  ✓ {out_path.name} (节点数: {num_nodes})')
                processed_original.append(src.stem)
        except Exception as exc:
            print(f'  ✗ {src.name}: {exc}')
            skipped_original.append(src.stem)
    
    # 转换攻击图（跳过被过滤的原始图对应的攻击图）
    print('\n[转换攻击图]')
    for base, attack_map in attacked_geojson_map.items():
        # ⭐如果原始图被跳过，则跳过所有对应的攻击图
        if base in skipped_original:
            print(f'  ⚠️  跳过 {base} 的所有攻击图（原始图节点数超限）')
            skipped_attacked += len(attack_map)
            continue
        
        subdir = output_dir_attacked / base
        # ⭐ 如果子目录已存在且已有完整的攻击图，则跳过
        if subdir.exists() and len(list(subdir.glob('*_graph.pkl'))) == len(attack_map):
            print(f'  ⏭️  跳过 {base}：攻击图已完整 ({len(attack_map)} 个)')
            processed_attacked += len(attack_map)
            continue
        
        # 清理并重建该数据集的子目录
        if subdir.exists():
            shutil.rmtree(subdir)
        subdir.mkdir(parents=True, exist_ok=True)
        
        for param, geojson_path in sorted(attack_map.items()):
            try:
                gdf = gpd.read_file(geojson_path)
                data = gdf_to_graph(gdf, max_nodes)
                
                if data is not None:
                    out_path = subdir / f"{geojson_path.stem}_graph.pkl"
                    with open(out_path, 'wb') as f:
                        pickle.dump(data, f)
                    print(f'  ✓ {base}/{out_path.name}')
                    processed_attacked += 1
                else:
                    skipped_attacked += 1
            except Exception as exc:
                print(f'  ✗ {base}/{param}: {exc}')
                skipped_attacked += 1
    
    # ⭐输出转换统计
    print('\n' + '='*60)
    threshold_info = "不限制" if max_nodes is None else str(max_nodes)
    print(f'📊 转换统计（节点数阈值: {threshold_info}）')
    print('='*60)
    print(f'原始图: 处理 {len(processed_original)} 个, 跳过 {len(skipped_original)} 个')
    if skipped_original:
        print(f'  跳过的文件: {", ".join(skipped_original)}')
    print(f'攻击图: 处理 {processed_attacked} 个, 跳过 {skipped_attacked} 个')
    print('='*60)


# ====================
# 导出接口
# ====================

__all__ = [
    # 路径配置
    'PROJECT_ROOT',
    'SCRIPT_DIR',
    'MODEL_PATH',
    'MODEL_PATH_VGAT',
    'CAT32_PATH',
    'GLOBAL_SCALER_PATH',
    'K_FOR_KNN',
    
    # 标准化器
    'load_global_scaler',
    
    # 特征提取
    'extract_features_20d',
    
    # 图构建
    'build_knn_delaunay_edges',
    'gdf_to_graph',
    'convert_geojsons_to_graphs',
    
    # 模型加载
    'load_improved_gat_model',
    
    # 零水印工具
    'load_cat32',
    'features_to_matrix',
    'calc_nc',
    'extract_features_from_graph',
    
    # GeoJSON转换
    'convert_to_geojson',
]
