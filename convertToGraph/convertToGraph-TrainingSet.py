#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""\n第二步：训练集图结构转换\n将vector_data和vector_data_attacked下的训练集矢量数据转换为GCN可处理的图结构\n使用无向K近邻图，k=8\n"""

import os
import sys
import geopandas as gpd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from sklearn.neighbors import kneighbors_graph
import shutil

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def get_env_knn_k(default=8):
    try:
        return max(1, int(os.environ.get("VGAT_KNN_K", str(default))))
    except Exception:
        return default


def get_graph_suffix():
    return os.environ.get("VGAT_GRAPH_SUFFIX", "").strip()


def get_max_attacks_per_class(default=0):
    try:
        return max(0, int(os.environ.get("VGAT_MAX_ATTACKS_PER_CLASS", str(default))))
    except Exception:
        return default


def get_resume_graph_build():
    return os.environ.get("VGAT_RESUME_GRAPH_BUILD", "0").strip().lower() in {"1", "true", "yes"}

class TrainSetVectorToGraphConverter:
    """训练集矢量数据转图结构转换器"""
    
    def __init__(self, vector_dir="../convertToGeoJson/GeoJson/TrainingSet", attacked_dir="../convertToGeoJson-Attacked/GeoJson-Attacked/TrainingSet", graph_dir=None):
        self.vector_dir = vector_dir
        self.attacked_dir = attacked_dir
        self.graph_dir = graph_dir or f"Graph/TrainingSet{get_graph_suffix()}"
        self.knn_k = get_env_knn_k()
        self.max_attacks_per_class = get_max_attacks_per_class()
        self.resume_graph_build = get_resume_graph_build()
        self.ensure_graph_dir()
        self.scaler = StandardScaler()
        self.scaler_fitted = False  # 标记scaler是否已经fit
    
    def ensure_graph_dir(self):
        """确保图数据目录存在"""
        if not os.path.exists(self.graph_dir):
            os.makedirs(self.graph_dir)
        # 统一使用首字母大写目录名
        os.makedirs(os.path.join(self.graph_dir, 'Original'), exist_ok=True)
        os.makedirs(os.path.join(self.graph_dir, 'Attacked'), exist_ok=True)
        # 创建cache目录用于存放标准化器
        os.makedirs(os.path.join(self.graph_dir, 'cache'), exist_ok=True)

    def clean_output_dirs(self):
        """清空输出目录，确保每次运行可完全替换"""
        original_path = os.path.join(self.graph_dir, 'Original')
        attacked_path = os.path.join(self.graph_dir, 'Attacked')

        # 清空 Original 下的文件
        if os.path.exists(original_path):
            for name in os.listdir(original_path):
                file_path = os.path.join(original_path, name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                else:
                    shutil.rmtree(file_path)
        else:
            os.makedirs(original_path, exist_ok=True)

        # 重新创建 Attacked 目录（删除整个目录以清理其所有子目录）
        if os.path.exists(attacked_path):
            shutil.rmtree(attacked_path)
        os.makedirs(attacked_path, exist_ok=True)
    
    def extract_features(self, geometry, row):
        """提取13维特征"""
        features = []
        
        # 1. 几何类型编码（3维）
        geom_type = geometry.geom_type if hasattr(geometry, 'geom_type') else 'Unknown'
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
        features.extend([0.0, 0.0])
        
        # 12-13. 根据几何类型调整面积和周长
        if geom_type == 'Point':
            # 点图层：面积和周长都为0
            features[-3] = 0.0  # 面积
            features[-2] = 0.0  # 周长
        elif geom_type in ['LineString', 'MultiLineString']:
            # 线图层：面积为0
            features[-3] = 0.0  # 面积
        
        return np.array(features, dtype=np.float32)
    
    def build_knn_graph(self, node_features, k=8):
        """构建无向K近邻图（适合GCN），k值自适应（最小为1）"""
        n_samples = len(node_features)
        
        # 特殊情况：只有1个节点，返回无边图
        if n_samples == 1:
            print(f"  📊 单节点图 (节点数=1, 无边)")
            return torch.empty((2, 0), dtype=torch.long)
        
        # 动态调整k值：最小为1，最大为n_samples-1
        actual_k = min(max(1, k), n_samples - 1)
        if actual_k < k:
            print(f"  📊 节点数{n_samples}<k={k}，自适应调整为k={actual_k}")
        else:
            print(f"  📊 构建无向K近邻图 (k={actual_k}, 节点数={n_samples})")
        
        # 使用质心坐标计算距离
        centroids = node_features[:, -2:]  # 最后两列是质心坐标
        
        # 确保centroids是2维数组
        if centroids.ndim == 1:
            centroids = centroids.reshape(1, -1)
        
        # 构建K近邻图
        adjacency_matrix = kneighbors_graph(centroids, n_neighbors=actual_k, mode='connectivity', include_self=False)
        
        # 转换为无向图：A_undirected = A + A^T
        adjacency_matrix = adjacency_matrix + adjacency_matrix.T
        
        # 转换为边索引格式
        edge_index = torch.tensor(np.vstack(adjacency_matrix.nonzero()), dtype=torch.long)
        
        return edge_index
    
    def fit_global_scaler(self):
        """第一遍扫描：用所有原始图的特征fit全局scaler"""
        print("\n📐 第一步：扫描所有原始图，构建全局标准化器...")
        all_features = []
        
        for filename in os.listdir(self.vector_dir):
            if filename.endswith('.geojson'):
                try:
                    file_path = os.path.join(self.vector_dir, filename)
                    gdf = gpd.read_file(file_path)
                    
                    # 提取特征
                    for idx, row in gdf.iterrows():
                        features = self.extract_features(row.geometry, row)
                        all_features.append(features)
                except Exception as e:
                    print(f"  ⚠️  扫描文件 {filename} 时出错: {e}")
                    continue
        
        if len(all_features) > 0:
            all_features = np.array(all_features, dtype=np.float32)
            self.scaler.fit(all_features)
            self.scaler_fitted = True
            print(f"  ✅ 全局标准化器已构建，使用 {len(all_features)} 个节点特征")
            print(f"  📊 特征均值: {self.scaler.mean_[:3]}... (前3维)")
            print(f"  📊 特征标准差: {self.scaler.scale_[:3]}... (前3维)")
            
            # 保存标准化器到cache目录
            scaler_path = os.path.join(self.graph_dir, 'cache', 'global_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"  💾 全局标准化器已保存至: {scaler_path}")
        else:
            print("  ⚠️  未找到有效特征数据")
    
    def build_graph_from_gdf(self, gdf, graph_name, use_fitted_scaler=True):
        """从GeoDataFrame构建图
        
        Args:
            gdf: GeoDataFrame
            graph_name: 图名称
            use_fitted_scaler: 是否使用已fit的scaler（True=transform only, False=fit_transform）
        """
        # 提取特征
        node_features = []
        for idx, row in gdf.iterrows():
            features = self.extract_features(row.geometry, row)
            node_features.append(features)
        
        node_features = np.array(node_features, dtype=np.float32)
        
        # 标准化特征：使用全局scaler
        if len(node_features) > 0:
            if use_fitted_scaler and self.scaler_fitted:
                node_features = self.scaler.transform(node_features)  # 只transform，不fit
                
                # ⭐ 特征裁剪已禁用
                # 原因：相对坐标特征已实现平移不变性，不需要裁剪
                #       裁剪会丢失区分信息，降低NC值25%
                # 结论：保持特征完整性 ✅
            else:
                node_features = self.scaler.fit_transform(node_features)  # fit + transform
        
        # 构建K近邻图 (k=8，自适应)
        edge_index = self.build_knn_graph(node_features, k=self.knn_k)
        
        # 创建PyTorch Geometric Data对象
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=edge_index
        )
        
        return data
    
    def save_graph_data(self, data, filename, subdir=None, data_type='Original'):
        """保存图数据，如果文件存在则覆盖
        
        Returns:
            bool: 保存成功返回True，失败返回False
        """
        try:
            if data_type == 'Original':
                # Original 文件夹下直接存放
                save_dir = os.path.join(self.graph_dir, 'Original')
            else:
                # Attacked 文件夹下使用原文件名作为子目录
                save_dir = os.path.join(self.graph_dir, 'Attacked', subdir if subdir else '')
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # 保存为pickle文件，如果存在则覆盖
            save_path = os.path.join(save_dir, f"{filename}_graph.pkl")
            temp_path = save_path + '.tmp'  # 使用临时文件
            
            # 先写入临时文件
            with open(temp_path, 'wb') as f:
                pickle.dump(data, f)
            
            # 验证文件完整性
            with open(temp_path, 'rb') as f:
                pickle.load(f)  # 尝试加载验证
            
            # 验证成功，重命名为正式文件
            if os.path.exists(save_path):
                os.remove(save_path)
            os.rename(temp_path, save_path)
            
            print(f"已保存: {save_path}")
            return True
            
        except Exception as e:
            print(f"  ❌ 保存文件失败: {e}")
            # 清理临时文件
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            return False
    
    def convert_train_set_to_graph(self):
        """转换训练集数据为图结构（全局标准化）"""
        print("="*60)
        print("🚀 开始转换训练集数据（使用全局标准化）")
        print("="*60)

        # 清理旧输出，确保可重复运行
        if not self.resume_graph_build:
            self.clean_output_dirs()
        else:
            print("\nℹ️  启用断点续跑：保留已生成的图数据并跳过已有文件")

        # 【关键】第一步：用所有原始图fit全局scaler
        cache_scaler_path = os.path.join(self.graph_dir, 'cache', 'global_scaler.pkl')
        if self.resume_graph_build and os.path.exists(cache_scaler_path):
            with open(cache_scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            self.scaler_fitted = True
            print(f"\n📐 第一步：复用已有全局标准化器 -> {cache_scaler_path}")
        else:
            self.fit_global_scaler()
        
        if not self.scaler_fitted:
            print("\n❌ 无法构建全局标准化器，终止转换")
            return

        # 处理原始数据
        print("\n🗂️  第二步：处理原始数据（TrainingSet/Original）...")
        original_stats = {"success": 0, "failed": 0, "skipped": 0}
        
        for filename in os.listdir(self.vector_dir):
            if filename.endswith('.geojson'):
                try:
                    # 读取geojson文件
                    file_path = os.path.join(self.vector_dir, filename)
                    gdf = gpd.read_file(file_path)
                    
                    # 验证数据有效性：只检查空文件（K值会自适应，最小为1）
                    if len(gdf) == 0:
                        print(f"  ⚠️  跳过空文件: {filename}")
                        original_stats["skipped"] += 1
                        continue
                    
                    graph_name = filename.replace('.geojson', '')
                    save_path = os.path.join(self.graph_dir, 'Original', f"{graph_name}_graph.pkl")
                    if self.resume_graph_build and os.path.exists(save_path):
                        print(f"  ⏭️  跳过已存在原始图 {graph_name}")
                        original_stats["skipped"] += 1
                        continue

                    # 构建图（使用全局scaler）
                    data = self.build_graph_from_gdf(gdf, filename, use_fitted_scaler=True)
                    
                    # 验证图数据有效性
                    if data.x.shape[0] == 0:
                        print(f"  ⚠️  跳过无效图数据: {filename} (节点数=0)")
                        original_stats["skipped"] += 1
                        continue
                    
                    # 保存到 Original（只有在完全成功后才保存）
                    if self.save_graph_data(data, graph_name, data_type='Original'):
                        print(f"  ✅ {graph_name}")
                        original_stats["success"] += 1
                    else:
                        print(f"  ❌ 保存失败，跳过: {filename}")
                        original_stats["failed"] += 1
                    
                except Exception as e:
                    print(f"  ❌ 处理文件 {filename} 时出错: {e}")
                    print(f"     跳过该文件，不生成图数据")
                    original_stats["failed"] += 1
                    continue
        
        print(f"\n📊 原始数据处理统计: 成功={original_stats['success']}, 失败={original_stats['failed']}, 跳过={original_stats['skipped']}")
        
        # 处理攻击数据（以原文件名为子目录）
        print("\n🗂️  第三步：处理攻击数据（TrainingSet/Attacked）...")
        attacked_stats = {"success": 0, "failed": 0, "skipped": 0}
        
        for attacked_subdir in sorted(os.listdir(self.attacked_dir)):
            attack_dir_path = os.path.join(self.attacked_dir, attacked_subdir)
            if os.path.isdir(attack_dir_path):
                print(f"\n📁 处理子目录: {attacked_subdir}")
                subdir_stats = {"success": 0, "failed": 0, "skipped": 0}
                
                attack_filenames = sorted(
                    filename for filename in os.listdir(attack_dir_path) if filename.endswith('.geojson')
                )
                attacked_graph_subdir = os.path.join(self.graph_dir, 'Attacked', attacked_subdir)
                existing_graph_count = 0
                if self.resume_graph_build and os.path.isdir(attacked_graph_subdir):
                    existing_graph_count = len(
                        [name for name in os.listdir(attacked_graph_subdir) if name.endswith('_graph.pkl')]
                    )

                target_count = len(attack_filenames)
                if self.max_attacks_per_class > 0:
                    target_count = min(target_count, self.max_attacks_per_class)

                if self.resume_graph_build and target_count > 0 and existing_graph_count >= target_count:
                    subdir_stats["skipped"] += existing_graph_count
                    attacked_stats["skipped"] += existing_graph_count
                    print(f"  Skip completed subdir: {attacked_subdir} ({existing_graph_count} existing graphs)")
                    print(f"  子目录统计: 成功={subdir_stats['success']}, 失败={subdir_stats['failed']}, 跳过={subdir_stats['skipped']}")
                    continue

                processed_in_subdir = existing_graph_count
                for filename in tqdm(attack_filenames, desc=f"处理 {attacked_subdir}"):
                    if filename.endswith('.geojson'):
                        graph_name = filename.replace('.geojson', '')
                        save_path = os.path.join(self.graph_dir, 'Attacked', attacked_subdir, f"{graph_name}_graph.pkl")
                        if self.resume_graph_build and os.path.exists(save_path):
                            subdir_stats["skipped"] += 1
                            attacked_stats["skipped"] += 1
                            if self.max_attacks_per_class > 0 and processed_in_subdir >= self.max_attacks_per_class:
                                print(f"  ℹ️  {attacked_subdir} 已有 {self.max_attacks_per_class} 个攻击图，停止继续转换")
                                break
                            continue
                        if self.max_attacks_per_class > 0 and processed_in_subdir >= self.max_attacks_per_class:
                            print(f"  ℹ️  {attacked_subdir} 已达到攻击样本上限 {self.max_attacks_per_class}，停止继续转换")
                            break
                        data = None  # 确保变量初始化
                        try:
                            # 读取文件
                            file_path = os.path.join(attack_dir_path, filename)
                            gdf = gpd.read_file(file_path)
                            
                            # 验证数据有效性：只检查空文件（K值会自适应，最小为1）
                            if len(gdf) == 0:
                                print(f"  ⚠️  跳过空文件: {filename}")
                                subdir_stats["skipped"] += 1
                                attacked_stats["skipped"] += 1
                                continue
                            
                            # 构建图（使用全局scaler）
                            data = self.build_graph_from_gdf(gdf, filename, use_fitted_scaler=True)
                            
                            # 验证图数据有效性
                            if data.x.shape[0] == 0:
                                print(f"  ⚠️  跳过无效图数据: {filename} (节点数=0)")
                                subdir_stats["skipped"] += 1
                                attacked_stats["skipped"] += 1
                                continue
                            
                            # 保存图数据（只有完全成功才保存）
                            if self.save_graph_data(data, graph_name, attacked_subdir, 'Attacked'):
                                subdir_stats["success"] += 1
                                attacked_stats["success"] += 1
                                processed_in_subdir += 1
                            else:
                                print(f"  ❌ 保存失败，跳过: {filename}")
                                subdir_stats["failed"] += 1
                                attacked_stats["failed"] += 1
                                
                        except Exception as e:
                            print(f"\n  ❌ 处理文件 {filename} 时出错: {e}")
                            print(f"     跳过该攻击，不生成图数据")
                            subdir_stats["failed"] += 1
                            attacked_stats["failed"] += 1
                            # 确保不保存任何部分数据
                            data = None
                            continue
                
                print(f"  📊 {attacked_subdir} 统计: 成功={subdir_stats['success']}, 失败={subdir_stats['failed']}, 跳过={subdir_stats['skipped']}")
        
        print(f"\n📊 攻击数据总体统计: 成功={attacked_stats['success']}, 失败={attacked_stats['failed']}, 跳过={attacked_stats['skipped']}")
        
        print("\n" + "="*60)
        print("✅ 训练集转换完成！")
        print("="*60)

def main():
    """主函数"""
    print("=== 训练集图结构转换 ===")
    
    # 创建转换器
    converter = TrainSetVectorToGraphConverter()
    
    # 转换训练集数据
    converter.convert_train_set_to_graph()
    
    print("训练集图结构转换完成！")

if __name__ == "__main__":
    main() 
