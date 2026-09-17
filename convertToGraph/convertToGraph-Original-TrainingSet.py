#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
只生成训练集的原始图（Original）
不处理Attacked图，用于快速恢复Original图数据
"""

import os
import geopandas as gpd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph

class OriginalGraphConverter:
    """只生成原始图的转换器"""
    
    def __init__(self, vector_dir="../convertToGeoJson/GeoJson/TrainingSet", graph_dir="Graph/TrainingSet"):
        self.vector_dir = vector_dir
        self.graph_dir = graph_dir
        self.ensure_graph_dir()
        self.scaler = StandardScaler()
        self.scaler_fitted = False
    
    def ensure_graph_dir(self):
        """确保图数据目录存在"""
        os.makedirs(self.graph_dir, exist_ok=True)
        os.makedirs(os.path.join(self.graph_dir, 'Original'), exist_ok=True)
        os.makedirs(os.path.join(self.graph_dir, 'cache'), exist_ok=True)
    
    def extract_features(self, geometry, row):
        """提取13维特征（与攻击图保持一致）"""
        features = []
        
        # 1. 几何类型编码（3维）
        geom_type = geometry.geom_type if hasattr(geometry, 'geom_type') else 'Unknown'
        if geom_type == 'Point':
            geom_features = [1, 0, 0]
        elif geom_type in ['LineString', 'MultiLineString']:
            geom_features = [0, 1, 0]
        elif geom_type in ['Polygon', 'MultiPolygon']:
            geom_features = [0, 0, 1]
        else:
            geom_features = [0, 0, 0]
        features.extend(geom_features)
        
        # 2. 面积
        features.append(geometry.area if hasattr(geometry, 'area') else 0.0)
        
        # 3. 周长
        features.append(geometry.length if hasattr(geometry, 'length') else 0.0)
        
        # 4-7. 边界框特征
        bounds = geometry.bounds
        features.extend([bounds[0], bounds[1], bounds[2], bounds[3]])
        
        # 8-9. 边界框尺寸
        features.extend([bounds[2] - bounds[0], bounds[3] - bounds[1]])
        
        # 10-11. 质心坐标
        centroid = geometry.centroid
        features.extend([centroid.x, centroid.y])
        
        # 12-13. 根据几何类型调整面积和周长（与旧脚本保持一致）
        if geom_type == 'Point':
            features[3] = 0.0  # 面积（索引3）
            features[4] = 0.0  # 周长（索引4）
        elif geom_type in ['LineString', 'MultiLineString']:
            features[3] = 0.0  # 面积（索引3）
        
        # ✅ 保持13维特征，与攻击图一致（不添加复杂度）
        return features
    
    def build_edge_index(self, node_features, k=8):
        """构建无向K近邻图的边索引"""
        n_samples = node_features.shape[0]
        
        if n_samples < 2:
            return torch.empty((2, 0), dtype=torch.long)
        
        # 动态调整k值
        actual_k = min(k, n_samples - 1)
        if actual_k < k:
            print(f"  📊 节点数{n_samples}<k={k}，自适应调整为k={actual_k}")
        else:
            print(f"  📊 构建无向K近邻图 (k={actual_k}, 节点数={n_samples})")
        
        # 使用质心坐标计算距离
        centroids = node_features[:, -2:]  # 最后两列是质心坐标
        
        if centroids.ndim == 1:
            centroids = centroids.reshape(1, -1)
        
        # 构建K近邻图
        adjacency_matrix = kneighbors_graph(centroids, n_neighbors=actual_k, mode='connectivity', include_self=False)
        
        # 转换为无向图
        adjacency_matrix = adjacency_matrix + adjacency_matrix.T
        
        # 转换为边索引格式
        edge_index = torch.tensor(np.vstack(adjacency_matrix.nonzero()), dtype=torch.long)
        
        return edge_index
    
    def fit_global_scaler(self):
        """扫描所有原始图，构建全局标准化器"""
        print("\n📐 第一步：扫描所有原始图，构建全局标准化器...")
        all_features = []
        
        if not os.path.exists(self.vector_dir):
            print(f"  ❌ 原始数据目录不存在: {self.vector_dir}")
            return
        
        geojson_files = [f for f in os.listdir(self.vector_dir) if f.endswith('.geojson')]
        if not geojson_files:
            print(f"  ❌ 未找到.geojson文件: {self.vector_dir}")
            return
        
        print(f"  发现 {len(geojson_files)} 个原始图文件")
        
        for filename in geojson_files:
            try:
                file_path = os.path.join(self.vector_dir, filename)
                gdf = gpd.read_file(file_path)
                
                # 提取特征
                for idx, row in gdf.iterrows():
                    features = self.extract_features(row.geometry, row)
                    all_features.append(features)
                    
                print(f"  ✅ {filename}: {len(gdf)} 个节点")
            except Exception as e:
                print(f"  ⚠️  扫描文件 {filename} 时出错: {e}")
                continue
        
        if len(all_features) > 0:
            all_features = np.array(all_features, dtype=np.float32)
            self.scaler.fit(all_features)
            self.scaler_fitted = True
            print(f"\n  ✅ 全局标准化器已构建，使用 {len(all_features)} 个节点特征")
            print(f"  📊 特征均值: {self.scaler.mean_[:3]}... (前3维)")
            print(f"  📊 特征标准差: {self.scaler.scale_[:3]}... (前3维)")
            
            # 保存标准化器
            scaler_path = os.path.join(self.graph_dir, 'global_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump({'scaler': self.scaler}, f)
            print(f"  💾 全局标准化器已保存至: {scaler_path}")
        else:
            print("  ⚠️  未找到有效特征数据")
    
    def build_graph_from_gdf(self, gdf, graph_name):
        """从GeoDataFrame构建图"""
        # 提取特征
        node_features = []
        for idx, row in gdf.iterrows():
            features = self.extract_features(row.geometry, row)
            node_features.append(features)
        
        node_features = np.array(node_features, dtype=np.float32)
        
        # 标准化特征：使用全局scaler
        if len(node_features) > 0:
            if self.scaler_fitted:
                node_features = self.scaler.transform(node_features)
            else:
                print(f"  ⚠️  警告: 全局scaler未fit，将使用局部标准化")
                node_features = self.scaler.fit_transform(node_features)
        
        # 构建边索引
        edge_index = self.build_edge_index(node_features, k=8)
        
        # 转换为PyTorch张量
        x = torch.tensor(node_features, dtype=torch.float32)
        
        # 创建Data对象
        data = Data(x=x, edge_index=edge_index)
        
        return data
    
    def save_graph_data(self, data, graph_name):
        """保存图数据到Original目录
        
        Returns:
            bool: 保存成功返回True，失败返回False
        """
        try:
            save_path = os.path.join(self.graph_dir, 'Original', f'{graph_name}_graph.pkl')
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
    
    def convert_original_only(self):
        """只转换原始图"""
        print("="*60)
        print("🚀 开始生成训练集原始图（Original Only）")
        print("="*60)
        
        # 第一步：构建全局scaler
        self.fit_global_scaler()
        
        if not self.scaler_fitted:
            print("\n❌ 无法构建全局标准化器，终止转换")
            return
        
        # 第二步：处理原始数据
        print("\n🗂️  第二步：处理原始数据（TrainingSet/Original）...")
        
        geojson_files = [f for f in os.listdir(self.vector_dir) if f.endswith('.geojson')]
        stats = {"success": 0, "failed": 0, "skipped": 0}
        
        for filename in geojson_files:
            try:
                # 读取geojson文件
                file_path = os.path.join(self.vector_dir, filename)
                gdf = gpd.read_file(file_path)
                
                # 验证数据有效性
                if len(gdf) == 0:
                    print(f"  ⚠️  跳过空文件: {filename}")
                    stats["skipped"] += 1
                    continue
                
                # 构建图
                data = self.build_graph_from_gdf(gdf, filename)
                
                # 验证图数据有效性
                if data.x.shape[0] == 0:
                    print(f"  ⚠️  跳过无效图数据: {filename} (节点数=0)")
                    stats["skipped"] += 1
                    continue
                
                # 生成文件名
                graph_name = filename.replace('.geojson', '')
                
                # 保存到Original（只有完全成功才保存）
                if self.save_graph_data(data, graph_name):
                    print(f"  ✅ {graph_name}: 节点={data.x.size(0)}, 边={data.edge_index.size(1)}")
                    stats["success"] += 1
                else:
                    print(f"  ❌ 保存失败，跳过: {filename}")
                    stats["failed"] += 1
                
            except Exception as e:
                print(f"  ❌ 处理文件 {filename} 时出错: {e}")
                print(f"     跳过该文件，不生成图数据")
                stats["failed"] += 1
                continue
        
        print(f"\n📊 处理统计: 成功={stats['success']}, 失败={stats['failed']}, 跳过={stats['skipped']}")
        success_count = stats["success"]
        fail_count = stats["failed"]
        
        print("\n" + "="*60)
        print(f"✅ 原始图生成完成！")
        print(f"  成功: {success_count} 个")
        print(f"  失败: {fail_count} 个")
        print("="*60)

def main():
    """主函数"""
    print("=== 训练集原始图生成器 ===\n")
    
    # 创建转换器
    converter = OriginalGraphConverter()
    
    # 只转换原始图
    converter.convert_original_only()
    
    print("\n✨ 原始图生成完成！现在可以运行训练了。")

if __name__ == "__main__":
    main()
