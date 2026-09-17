#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第一步：生成被攻击的矢量数据
按照attack200.py的逻辑：前100个指定攻击方式，后100个随机组合攻击
为每个图生成200个被攻击的矢量地图类型放入vector_data_attacked的各个图的子文件夹中
"""

import os
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon  # 保留以支持各种几何类型
from shapely.affinity import rotate, scale, translate
from shapely.ops import split as shp_split
from shapely.geometry import LineString as ShpLineString
import random
from tqdm import tqdm
import shutil
import math

class VectorAttackGenerator:
    """矢量数据攻击生成器"""
    
    def __init__(self, input_dir="../convertToGeoJson/GeoJson/TrainingSet", output_dir="GeoJson-Attacked/TrainingSet"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.ensure_output_dir()
        
        # ✅ 优化：定义120个单体攻击方式（针对低NC攻击增加样本）
        self.single_attacks = []
        
        # Fig1: 删除顶点 - 15个（10%-90% + 额外高比例变体）
        for pct in range(10, 100, 10):  # 10%, 20%, ..., 90% (9个)
            self.single_attacks.append((f"delete_{pct}pct_vertices.geojson", f"删除{pct}%顶点"))
        # 额外增加85%删除的变体 (6个)
        for i in range(1, 7):
            self.single_attacks.append((f"delete_85pct_vertices_v{i}.geojson", f"删除85%顶点_变体{i}"))
        
        # Fig2: 添加顶点 - ✅ 30个（重点增强，解决低NC问题）
        # 强度0,1,2 × 比例20%,40%,60%,80% (12个)
        for strength in [0, 1, 2]:
            for pct in [20, 40, 60, 80]:
                self.single_attacks.append((f"add_strength{strength}_{pct}pct_vertices.geojson", f"添加{pct}%顶点_强度{strength}"))
        # 额外增加强度1和2的更多比例组合 (12个)
        for strength in [1, 2]:
            for pct in [15, 35, 55, 75, 85, 95]:
                self.single_attacks.append((f"add_strength{strength}_{pct}pct_vertices_extra.geojson", f"添加{pct}%顶点_强度{strength}_额外"))
        # 再增加6个强度1+50%的变体（最弱区域）
        for i in range(1, 7):
            self.single_attacks.append((f"add_strength1_50pct_vertices_v{i}.geojson", f"添加50%顶点_强度1_变体{i}"))
        
        # Fig3: 删除对象 - 10个（10%-90%）
        for pct in range(10, 100, 10):  # 10%, 20%, ..., 90% (9个)
            self.single_attacks.append((f"delete_{pct}pct_objects.geojson", f"删除{pct}%对象"))
        # 额外增加1个50%删除的变体
        self.single_attacks.append(("delete_50pct_objects_v1.geojson", "删除50%对象_变体1"))
        
        # Fig4: 噪声攻击 - ✅ 30个（重点增强，解决低NC问题）
        # 基础：强度[0.4, 0.6, 0.8] × 比例[20, 40, 60, 80] (12个)
        for strength in [0.4, 0.6, 0.8]:
            for pct in [20, 40, 60, 80]:
                self.single_attacks.append((f"noise_{pct}pct_strength_{strength}.geojson", f"噪声{pct}%顶点_强度{strength}"))
        # 额外增加强度0.5和0.7的样本 (12个)
        for strength in [0.5, 0.7]:
            for pct in [20, 30, 50, 70, 80, 90]:
                self.single_attacks.append((f"noise_{pct}pct_strength_{strength}_extra.geojson", f"噪声{pct}%顶点_强度{strength}_额外"))
        # 再增加6个高强度50%的变体（最弱区域）
        for i in range(1, 7):
            self.single_attacks.append((f"noise_50pct_strength_0.8_v{i}.geojson", f"噪声50%顶点_强度0.8_变体{i}"))
        
        # Fig5: 裁剪 - 8个
        self.single_attacks.extend([
            ("crop_x_center_50pct.geojson", "沿X轴中心裁剪50%"),
            ("crop_y_center_50pct.geojson", "沿Y轴中心裁剪50%"),
            ("crop_top_left.geojson", "裁剪左上角区域"),
            ("crop_bottom_right.geojson", "裁剪右下角区域"),
            ("crop_random_40pct.geojson", "随机裁剪40%"),
            ("crop_center_30pct.geojson", "中心裁剪30%"),
            ("crop_edge_20pct.geojson", "边缘裁剪20%"),
            ("crop_diagonal.geojson", "对角线裁剪"),
        ])
        
        # Fig6: 平移 - 5个
        self.single_attacks.extend([
            ("translate_x_20.geojson", "沿X轴平移20"),
            ("translate_y_20.geojson", "沿Y轴平移20"),
            ("translate_20_20.geojson", "X、Y轴各平移20"),
            ("translate_20_40.geojson", "X轴平移20_Y轴平移40"),
            ("translate_30_10.geojson", "X轴平移30_Y轴平移10"),
        ])
        
        # Fig7: 缩放 - 6个
        for factor in [0.5, 0.7, 0.9, 1.2, 1.5, 2.0]:
            pct = int(round(factor * 100))
            self.single_attacks.append((f"scale_{pct}pct.geojson", f"缩放{factor}倍"))
        
        # Fig8: 旋转 - 8个
        for deg in [45, 90, 135, 180, 225, 270, 315, 360]:
            self.single_attacks.append((f"rotate_{deg}deg.geojson", f"旋转{deg}度"))
        
        # Fig9: 翻转 - 3个
        self.single_attacks.extend([
            ("flip_x.geojson", "X轴镜像翻转"),
            ("flip_y.geojson", "Y轴镜像翻转"),
            ("flip_xy.geojson", "XY轴同时翻转"),
        ])
        
        # Fig10: 打乱顺序 - 5个
        self.single_attacks.extend([
            ("reverse_vertices.geojson", "反转顶点顺序"),
            ("reverse_objects.geojson", "反转对象顺序"),
            ("shuffle_objects.geojson", "打乱对象顺序"),
            ("shuffle_vertices_v1.geojson", "打乱顶点顺序_v1"),
            ("shuffle_vertices_v2.geojson", "打乱顶点顺序_v2"),
        ])
        
        print(f"✅ 共定义{len(self.single_attacks)}个单体攻击")
        
        # ✅ 优化：80个组合攻击（增强对Fig12复合攻击的鲁棒性）
        self.combo_attacks = []
        
        # 1. 全攻击链（Fig12风格）- 1个
        self.combo_attacks.append(("combo_full_attack_chain.geojson", "全攻击链(Fig1→Fig10顺序)"))
        
        # 2. 重度组合攻击（6-8种攻击）- 20个
        for i in range(1, 21):
            self.combo_attacks.append((f"combo_heavy_{i:02d}.geojson", f"重度组合{i}(6-8种)"))
        
        # 3. 中度组合攻击（4-5种攻击）- 30个
        for i in range(1, 31):
            self.combo_attacks.append((f"combo_medium_{i:02d}.geojson", f"中度组合{i}(4-5种)"))
        
        # 4. 轻度组合攻击（2-3种攻击）- 29个
        for i in range(1, 30):
            self.combo_attacks.append((f"combo_light_{i:02d}.geojson", f"轻度组合{i}(2-3种)"))
        
        print(f"✅ 共定义{len(self.combo_attacks)}个组合攻击")
        print(f"✅ 总计{len(self.single_attacks) + len(self.combo_attacks)}个攻击样本")
    
    def ensure_output_dir(self):
        """确保输出目录存在"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def load_vector_data(self, filename):
        """加载矢量数据"""
        filepath = os.path.join(self.input_dir, filename)
        try:
            gdf = gpd.read_file(filepath)
            print(f"成功加载矢量数据: {filename}")
            print(f"数据包含 {len(gdf)} 个要素")
            return gdf
        except Exception as e:
            print(f"加载数据失败: {e}")
            return None
    
    def apply_delete_vertices_attack(self, gdf, percentage):
        """删除指定百分比的顶点"""
        def delete_vertices_from_geom(geom, pct):
            if isinstance(geom, LineString):
                coords = list(geom.coords)
                if len(coords) <= 2:
                    return geom
                n_to_delete = max(1, int((len(coords) - 2) * pct / 100))
                if n_to_delete >= len(coords) - 2:
                    return geom
                indices = list(range(1, len(coords) - 1))
                to_delete = set(random.sample(indices, n_to_delete))
                new_coords = [coords[0]] + [coords[i] for i in range(1, len(coords) - 1) if i not in to_delete] + [coords[-1]]
                return LineString(new_coords)
            elif isinstance(geom, Polygon):
                ext_coords = list(geom.exterior.coords)
                if len(ext_coords) <= 4:
                    return geom
                n_to_delete = max(1, int((len(ext_coords) - 4) * pct / 100))
                if n_to_delete >= len(ext_coords) - 4:
                    return geom
                indices = list(range(1, len(ext_coords) - 2))
                to_delete = set(random.sample(indices, n_to_delete))
                new_ext_coords = [ext_coords[0]] + [ext_coords[i] for i in range(1, len(ext_coords) - 2) if i not in to_delete] + [ext_coords[-2], ext_coords[-1]]
                holes = []
                for ring in geom.interiors:
                    hole_coords = list(ring.coords)
                    if len(hole_coords) > 4:
                        n_to_delete_hole = max(1, int((len(hole_coords) - 4) * pct / 100))
                        if n_to_delete_hole < len(hole_coords) - 4:
                            indices_hole = list(range(1, len(hole_coords) - 2))
                            to_delete_hole = set(random.sample(indices_hole, n_to_delete_hole))
                            new_hole_coords = [hole_coords[0]] + [hole_coords[i] for i in range(1, len(hole_coords) - 2) if i not in to_delete_hole] + [hole_coords[-2], hole_coords[-1]]
                            holes.append(new_hole_coords)
                        else:
                            holes.append(hole_coords)
                    else:
                        holes.append(hole_coords)
                return Polygon(new_ext_coords, holes=holes if holes else None)
            return geom
        
        gdf_attacked = gdf.copy()
        gdf_attacked['geometry'] = gdf_attacked['geometry'].apply(
            lambda geom: delete_vertices_from_geom(geom, percentage)
        )
        return gdf_attacked
    
    def apply_delete_objects_attack(self, gdf, percentage):
        """删除指定百分比的对象"""
        gdf_attacked = gdf.copy()
        num_objects = len(gdf_attacked)
        num_to_delete = int(num_objects * percentage / 100)
        if num_to_delete > 0:
            indices_to_delete = random.sample(range(num_objects), num_to_delete)
            gdf_attacked = gdf_attacked.drop(indices_to_delete).reset_index(drop=True)
        return gdf_attacked
    
    def apply_add_vertices_attack(self, gdf, percentage, noise_strength=0.0):
        """✅ 添加指定百分比的顶点（支持噪声强度）
        
        Args:
            gdf: GeoDataFrame
            percentage: 添加比例
            noise_strength: 噪声强度 (0=无噪声, 1=小噪声, 2=中噪声)
        """
        # 噪声标准差映射
        noise_sigma_map = {0: 0.0, 1: 0.01, 2: 0.03}
        noise_sigma = noise_sigma_map.get(noise_strength, 0.0)
        
        def add_vertices_to_geom(geom, pct, sigma):
            if isinstance(geom, LineString):
                coords = list(geom.coords)
                if len(coords) < 2:
                    return geom
                # 限制添加的顶点数量，避免过度复杂化
                n_to_add = min(3, max(1, int((len(coords) - 1) * pct / 100)))
                new_coords = [coords[0]]
                for i in range(len(coords) - 1):
                    p1, p2 = coords[i], coords[i + 1]
                    new_coords.append(p1)
                    for j in range(n_to_add):
                        t = (j + 1) / (n_to_add + 1)
                        mid_x = p1[0] + t * (p2[0] - p1[0])
                        mid_y = p1[1] + t * (p2[1] - p1[1])
                        # ✅ 添加噪声
                        if sigma > 0:
                            mid_x += np.random.normal(0, sigma)
                            mid_y += np.random.normal(0, sigma)
                        new_coords.append((mid_x, mid_y))
                new_coords.append(coords[-1])
                return LineString(new_coords)
            elif isinstance(geom, Polygon):
                ext_coords = list(geom.exterior.coords)
                if len(ext_coords) < 4:
                    return geom
                # 限制添加的顶点数量，避免过度复杂化
                n_to_add = min(3, max(1, int((len(ext_coords) - 1) * pct / 100)))
                new_ext_coords = [ext_coords[0]]
                for i in range(len(ext_coords) - 1):
                    p1, p2 = ext_coords[i], ext_coords[i + 1]
                    new_ext_coords.append(p1)
                    for j in range(n_to_add):
                        t = (j + 1) / (n_to_add + 1)
                        mid_x = p1[0] + t * (p2[0] - p1[0])
                        mid_y = p1[1] + t * (p2[1] - p1[1])
                        # ✅ 添加噪声
                        if sigma > 0:
                            mid_x += np.random.normal(0, sigma)
                            mid_y += np.random.normal(0, sigma)
                        new_ext_coords.append((mid_x, mid_y))
                new_ext_coords.append(ext_coords[-1])
                holes = []
                for ring in geom.interiors:
                    ring_coords = list(ring.coords)
                    if len(ring_coords) >= 4:
                        new_ring_coords = [ring_coords[0]]
                        for i in range(len(ring_coords) - 1):
                            p1, p2 = ring_coords[i], ring_coords[i + 1]
                            new_ring_coords.append(p1)
                            for j in range(n_to_add):
                                t = (j + 1) / (n_to_add + 1)
                                mid_x = p1[0] + t * (p2[0] - p1[0])
                                mid_y = p1[1] + t * (p2[1] - p1[1])
                                # ✅ 添加噪声
                                if sigma > 0:
                                    mid_x += np.random.normal(0, sigma)
                                    mid_y += np.random.normal(0, sigma)
                                new_ring_coords.append((mid_x, mid_y))
                        new_ring_coords.append(ring_coords[-1])
                        holes.append(new_ring_coords)
                    else:
                        holes.append(ring_coords)
                return Polygon(new_ext_coords, holes=holes if holes else None)
            return geom
        
        gdf_attacked = gdf.copy()
        gdf_attacked['geometry'] = gdf_attacked['geometry'].apply(
            lambda geom: add_vertices_to_geom(geom, percentage, noise_sigma)
        )
        return gdf_attacked
    
    def apply_noise_attack(self, gdf, percentage, strength):
        """噪声扰动攻击 - 顶点级扰动"""
        def jitter_vertices(geom, pct, strength):
            if isinstance(geom, LineString):
                coords = list(geom.coords)
                n = len(coords)
                k = max(1, int(n * pct / 100))
                indices = list(range(n))
                chosen = set(random.sample(indices, min(k, len(indices))))
                new_coords = []
                for i, coord in enumerate(coords):
                    if i in chosen:
                        new_coords.append((
                            coord[0] + random.uniform(-strength, strength),
                            coord[1] + random.uniform(-strength, strength)
                        ))
                    else:
                        new_coords.append(coord)
                return LineString(new_coords)
            elif isinstance(geom, Polygon):
                ext_coords = list(geom.exterior.coords)
                n = len(ext_coords)
                k = max(1, int(n * pct / 100))
                indices = list(range(n))
                chosen = set(random.sample(indices, min(k, len(indices))))
                new_ext_coords = []
                for i, coord in enumerate(ext_coords):
                    if i in chosen:
                        new_ext_coords.append((
                            coord[0] + random.uniform(-strength, strength),
                            coord[1] + random.uniform(-strength, strength)
                        ))
                    else:
                        new_ext_coords.append(coord)
                holes = []
                for ring in geom.interiors:
                    ring_coords = list(ring.coords)
                    n_ring = len(ring_coords)
                    k_ring = max(1, int(n_ring * pct / 100))
                    indices_ring = list(range(n_ring))
                    chosen_ring = set(random.sample(indices_ring, min(k_ring, len(indices_ring))))
                    new_ring_coords = []
                    for i, coord in enumerate(ring_coords):
                        if i in chosen_ring:
                            new_ring_coords.append((
                                coord[0] + random.uniform(-strength, strength),
                                coord[1] + random.uniform(-strength, strength)
                            ))
                        else:
                            new_ring_coords.append(coord)
                    holes.append(new_ring_coords)
                return Polygon(new_ext_coords, holes=holes if holes else None)
            return geom
        
        gdf_attacked = gdf.copy()
        gdf_attacked['geometry'] = gdf_attacked['geometry'].apply(
            lambda geom: jitter_vertices(geom, percentage, strength)
        )
        return gdf_attacked
    
    def apply_crop_attack(self, gdf, crop_type):
        """裁剪攻击"""
        gdf_attacked = gdf.copy()
        bounds = gdf_attacked.total_bounds
        bdf = gdf_attacked.geometry.bounds  # DataFrame: minx, miny, maxx, maxy
        
        if crop_type == "x_center_50pct":
            # 沿X轴中心裁剪50%
            mid_x = (bounds[0] + bounds[2]) / 2
            gdf_attacked = gdf_attacked[bdf['minx'] < mid_x].reset_index(drop=True)
        elif crop_type == "y_center_50pct":
            # 沿Y轴中心裁剪50%
            mid_y = (bounds[1] + bounds[3]) / 2
            gdf_attacked = gdf_attacked[bdf['miny'] < mid_y].reset_index(drop=True)
        elif crop_type == "top_left":
            # 裁剪左上角区域
            gdf_attacked = gdf_attacked[
                (bdf['minx'] < (bounds[0] + bounds[2]) / 2) &
                (bdf['miny'] > (bounds[1] + bounds[3]) / 2)
            ].reset_index(drop=True)
        elif crop_type == "bottom_right":
            # 裁剪右下角区域
            gdf_attacked = gdf_attacked[
                (bdf['minx'] > (bounds[0] + bounds[2]) / 2) &
                (bdf['miny'] < (bounds[1] + bounds[3]) / 2)
            ].reset_index(drop=True)
        elif crop_type == "random_40pct":
            # 随机裁剪40%
            num_objects = len(gdf_attacked)
            num_to_keep = int(num_objects * 0.6)
            if num_to_keep > 0:
                indices_to_keep = random.sample(range(num_objects), num_to_keep)
                gdf_attacked = gdf_attacked.iloc[indices_to_keep].reset_index(drop=True)
        elif crop_type == "center_30pct":
            # ✅ 中心裁剪30%（保留中心70%）
            mid_x = (bounds[0] + bounds[2]) / 2
            mid_y = (bounds[1] + bounds[3]) / 2
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            crop_width = width * 0.15
            crop_height = height * 0.15
            gdf_attacked = gdf_attacked[
                (bdf['minx'] > mid_x - crop_width) &
                (bdf['maxx'] < mid_x + crop_width) &
                (bdf['miny'] > mid_y - crop_height) &
                (bdf['maxy'] < mid_y + crop_height)
            ].reset_index(drop=True)
        elif crop_type == "edge_20pct":
            # ✅ 边缘裁剪20%（移除最外围20%）
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            margin_x = width * 0.2
            margin_y = height * 0.2
            gdf_attacked = gdf_attacked[
                (bdf['minx'] > bounds[0] + margin_x) &
                (bdf['maxx'] < bounds[2] - margin_x) &
                (bdf['miny'] > bounds[1] + margin_y) &
                (bdf['maxy'] < bounds[3] - margin_y)
            ].reset_index(drop=True)
        elif crop_type == "diagonal":
            # ✅ 对角线裁剪（保留对角线一侧）
            mid_x = (bounds[0] + bounds[2]) / 2
            mid_y = (bounds[1] + bounds[3]) / 2
            # 保留左下+右上区域
            gdf_attacked = gdf_attacked[
                ((bdf['minx'] < mid_x) & (bdf['miny'] < mid_y)) |
                ((bdf['minx'] > mid_x) & (bdf['miny'] > mid_y))
            ].reset_index(drop=True)
        
        return gdf_attacked
    
    def apply_translate_attack(self, gdf, dx, dy):
        """平移攻击"""
        gdf_attacked = gdf.copy()
        gdf_attacked['geometry'] = gdf_attacked['geometry'].apply(
            lambda geom: translate(geom, dx, dy)
        )
        return gdf_attacked
    
    def apply_scale_attack(self, gdf, scale_x, scale_y=None):
        """缩放攻击（使用全局中心）"""
        gdf_attacked = gdf.copy()
        if scale_y is None:
            scale_y = scale_x
        
        # ✅ 使用全局中心作为变换原点，与Fig8/Fig9保持一致
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        global_center = ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)
        
        gdf_attacked['geometry'] = gdf_attacked['geometry'].apply(
            lambda geom: scale(geom, scale_x, scale_y, origin=global_center)
        )
        return gdf_attacked
    
    def apply_rotate_attack(self, gdf, angle):
        """旋转攻击（使用全局中心）"""
        gdf_attacked = gdf.copy()
        
        # ✅ 使用全局中心作为旋转原点，与Fig8/Fig9保持一致
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        global_center = ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)
        
        gdf_attacked['geometry'] = gdf_attacked['geometry'].apply(
            lambda geom: rotate(geom, angle, origin=global_center)
        )
        return gdf_attacked
    
    def apply_flip_attack(self, gdf, flip_type):
        """翻转攻击（使用全局中心）"""
        gdf_attacked = gdf.copy()
        
        # ✅ 使用全局中心作为翻转原点，与Fig8/Fig9保持一致
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        global_center = ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)
        
        if flip_type == "x":
            gdf_attacked['geometry'] = gdf_attacked['geometry'].apply(
                lambda geom: scale(geom, -1, 1, origin=global_center)
            )
        elif flip_type == "y":
            gdf_attacked['geometry'] = gdf_attacked['geometry'].apply(
                lambda geom: scale(geom, 1, -1, origin=global_center)
            )
        elif flip_type == "xy":
            # ✅ 与Fig9保持一致：双轴翻转使用旋转180°实现（数学等价，更稳定）
            gdf_attacked['geometry'] = gdf_attacked['geometry'].apply(
                lambda geom: rotate(geom, 180, origin=global_center)
            )
        return gdf_attacked
    
    def apply_reverse_vertices_attack(self, gdf):
        """反转顶点顺序攻击"""
        def reverse_vertices(geom):
            if isinstance(geom, LineString):
                return LineString(list(geom.coords)[::-1])
            elif isinstance(geom, Polygon):
                ext_coords = list(geom.exterior.coords)
                ext_coords = ext_coords[:-1][::-1] + [ext_coords[0]]
                holes = []
                for ring in geom.interiors:
                    ring_coords = list(ring.coords)
                    ring_coords = ring_coords[:-1][::-1] + [ring_coords[0]]
                    holes.append(ring_coords)
                return Polygon(ext_coords, holes=holes if holes else None)
            return geom
        
        gdf_attacked = gdf.copy()
        gdf_attacked['geometry'] = gdf_attacked['geometry'].apply(reverse_vertices)
        return gdf_attacked
    
    def apply_shuffle_vertices_attack(self, gdf):
        """打乱顶点顺序攻击"""
        def shuffle_vertices(geom):
            if isinstance(geom, LineString):
                coords = list(geom.coords)
                if len(coords) <= 2:
                    return geom
                core = coords[1:-1]
                random.shuffle(core)
                return LineString([coords[0]] + core + [coords[-1]])
            elif isinstance(geom, Polygon):
                ext_coords = list(geom.exterior.coords)
                if len(ext_coords) <= 4:
                    return geom
                core = ext_coords[1:-2]
                random.shuffle(core)
                new_ext_coords = [ext_coords[0]] + core + [ext_coords[-2], ext_coords[-1]]
                holes = []
                for ring in geom.interiors:
                    ring_coords = list(ring.coords)
                    if len(ring_coords) > 4:
                        core_ring = ring_coords[1:-2]
                        random.shuffle(core_ring)
                        new_ring_coords = [ring_coords[0]] + core_ring + [ring_coords[-2], ring_coords[-1]]
                        holes.append(new_ring_coords)
                    else:
                        holes.append(ring_coords)
                return Polygon(new_ext_coords, holes=holes if holes else None)
            return geom
        
        gdf_attacked = gdf.copy()
        gdf_attacked['geometry'] = gdf_attacked['geometry'].apply(shuffle_vertices)
        return gdf_attacked
    
    def apply_merge_objects_attack(self, gdf):
        """合并对象攻击"""
        if len(gdf) < 2:
            return gdf.copy()
        
        gdf_attacked = gdf.copy()
        indices = list(range(len(gdf_attacked)))
        random.shuffle(indices)
        
        merged_geoms = []
        used = set()
        
        for i in range(0, len(indices) - 1, 2):
            idx1, idx2 = indices[i], indices[i + 1]
            try:
                geom1 = gdf_attacked.geometry.iloc[idx1]
                geom2 = gdf_attacked.geometry.iloc[idx2]
                merged = geom1.union(geom2)
                merged_geoms.append(merged)
                used.add(idx1)
                used.add(idx2)
            except Exception:
                pass
        
        # 保留未合并的对象
        remaining_geoms = [gdf_attacked.geometry.iloc[i] for i in range(len(gdf_attacked)) if i not in used]
        
        # 创建新的GeoDataFrame
        new_gdf = gdf_attacked.iloc[:0].copy()
        new_gdf['geometry'] = None
        new_gdf = new_gdf.reindex(range(len(remaining_geoms) + len(merged_geoms)))
        new_gdf['geometry'] = remaining_geoms + merged_geoms
        new_gdf = new_gdf.reset_index(drop=True)
        
        return new_gdf
    
    def apply_split_objects_attack(self, gdf):
        """拆分对象攻击"""
        def split_polygon(geom):
            if isinstance(geom, Polygon):
                bounds = geom.bounds
                cx = (bounds[0] + bounds[2]) / 2
                cy = (bounds[1] + bounds[3]) / 2
                length = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 2
                
                # 随机选择切割角度
                angle = random.uniform(0, math.pi)
                dx = math.cos(angle) * length
                dy = math.sin(angle) * length
                
                cutter = ShpLineString([(cx - dx, cy - dy), (cx + dx, cy + dy)])
                
                try:
                    parts = shp_split(geom, cutter)
                    return list(parts.geoms)
                except Exception:
                    return [geom]
            return [geom]
        
        gdf_attacked = gdf.copy()
        new_geoms = []
        
        for geom in gdf_attacked.geometry:
            if random.random() < 0.5:  # 50%概率进行拆分
                split_parts = split_polygon(geom)
                new_geoms.extend(split_parts)
            else:
                new_geoms.append(geom)
        
        # 创建新的GeoDataFrame
        new_gdf = gdf_attacked.iloc[:0].copy()
        new_gdf['geometry'] = None
        new_gdf = new_gdf.reindex(range(len(new_geoms)))
        new_gdf['geometry'] = new_geoms
        new_gdf = new_gdf.reset_index(drop=True)
        
        return new_gdf
    
    def apply_single_attack(self, gdf, attack_name):
        """✅ 优化：应用单体攻击（使用正则表达式模式匹配）"""
        import re
        
        # 删除顶点攻击 (支持所有百分比)
        match = re.search(r'delete_(\d+)pct_vertices', attack_name)
        if match:
            return self.apply_delete_vertices_attack(gdf, int(match.group(1)))
        
        # 删除对象攻击 (支持所有百分比)
        match = re.search(r'delete_(\d+)pct_objects', attack_name)
        if match:
            return self.apply_delete_objects_attack(gdf, int(match.group(1)))
        
        # 添加顶点攻击（支持噪声强度）
        match = re.search(r'add_strength(\d+)_(\d+)pct_vertices', attack_name)
        if match:
            strength = int(match.group(1))
            pct = int(match.group(2))
            return self.apply_add_vertices_attack(gdf, pct, noise_strength=strength)
        
        # 噪声攻击 (支持所有强度和比例)
        match = re.search(r'noise_(\d+)pct_strength_([0-9.]+)', attack_name)
        if match:
            pct = int(match.group(1))
            strength = float(match.group(2))
            return self.apply_noise_attack(gdf, pct, strength)
        
        # 裁剪攻击
        crop_types = {
            "crop_x_center_50pct": "x_center_50pct",
            "crop_y_center_50pct": "y_center_50pct",
            "crop_top_left": "top_left",
            "crop_bottom_right": "bottom_right",
            "crop_random_40pct": "random_40pct",
            "crop_center_30pct": "center_30pct",
            "crop_edge_20pct": "edge_20pct",
            "crop_diagonal": "diagonal"
        }
        for key, crop_type in crop_types.items():
            if key in attack_name:
                return self.apply_crop_attack(gdf, crop_type)
        
        # 平移攻击
        if "translate_x_20" in attack_name and "translate_20" not in attack_name:
            return self.apply_translate_attack(gdf, 20, 0)
        elif "translate_y_20" in attack_name and "translate_20" not in attack_name:
            return self.apply_translate_attack(gdf, 0, 20)
        elif "translate_20_20" in attack_name:
            return self.apply_translate_attack(gdf, 20, 20)
        elif "translate_20_40" in attack_name:
            return self.apply_translate_attack(gdf, 20, 40)
        elif "translate_30_10" in attack_name:
            return self.apply_translate_attack(gdf, 30, 10)
        
        # 缩放攻击 (支持所有比例)
        match = re.search(r'scale_(\d+)pct', attack_name)
        if match:
            factor = int(match.group(1)) / 100.0
            return self.apply_scale_attack(gdf, factor)
        
        # 旋转攻击 (支持所有角度)
        match = re.search(r'rotate_(\d+)deg', attack_name)
        if match:
            angle = int(match.group(1))
            return self.apply_rotate_attack(gdf, angle)
        
        # 翻转攻击
        if "flip_xy" in attack_name:
            return self.apply_flip_attack(gdf, "xy")
        elif "flip_x" in attack_name:
            return self.apply_flip_attack(gdf, "x")
        elif "flip_y" in attack_name:
            return self.apply_flip_attack(gdf, "y")
        
        # 顺序攻击
        if "reverse_vertices" in attack_name:
            return self.apply_reverse_vertices_attack(gdf)
        elif "reverse_objects" in attack_name:
            return gdf.iloc[::-1].reset_index(drop=True)
        elif "shuffle_vertices" in attack_name:
            return self.apply_shuffle_vertices_attack(gdf)
        elif "shuffle_objects" in attack_name:
            return gdf.sample(frac=1, random_state=random.randint(0, 10000)).reset_index(drop=True)
        
        # 未匹配的攻击类型
        print(f"  ⚠️ 未识别的攻击类型: {attack_name}，返回原图")
        return gdf.copy()
    
    def apply_combo_attack(self, gdf, attack_name):
        """✅ 优化：应用组合攻击（支持全攻击链和不同强度）"""
        gdf_attacked = gdf.copy()
        
        # 全攻击链（Fig12风格）
        if "full_attack_chain" in attack_name:
            # 按Fig1→Fig10顺序依次应用10种攻击
            gdf_attacked = self.apply_delete_vertices_attack(gdf_attacked, 10)
            gdf_attacked = self.apply_add_vertices_attack(gdf_attacked, 50, noise_strength=1)
            gdf_attacked = self.apply_delete_objects_attack(gdf_attacked, 50)
            gdf_attacked = self.apply_noise_attack(gdf_attacked, 50, 0.8)
            gdf_attacked = self.apply_crop_attack(gdf_attacked, "y_center_50pct")
            gdf_attacked = self.apply_translate_attack(gdf_attacked, 20, 40)
            gdf_attacked = self.apply_scale_attack(gdf_attacked, 0.9)
            gdf_attacked = self.apply_rotate_attack(gdf_attacked, 180)
            gdf_attacked = self.apply_flip_attack(gdf_attacked, "y")
            gdf_attacked = self.apply_reverse_vertices_attack(gdf_attacked)
            gdf_attacked = gdf_attacked.iloc[::-1].reset_index(drop=True)
            return gdf_attacked
        
        # 重度组合攻击（6-8种攻击）
        if "combo_heavy" in attack_name:
            num_attacks = random.randint(6, 8)
        # 中度组合攻击（4-5种攻击）
        elif "combo_medium" in attack_name:
            num_attacks = random.randint(4, 5)
        # 轻度组合攻击（2-3种攻击）
        elif "combo_light" in attack_name:
            num_attacks = random.randint(2, 3)
        else:
            num_attacks = random.randint(2, 4)
        
        # ✅ 扩展攻击类型池（增加add_vertices和更多噪声）
        attack_types = ['translate', 'rotate', 'scale', 'noise', 'crop', 'flip', 'add_vertices', 'delete_vertices']
        
        for _ in range(num_attacks):
            attack_type = random.choice(attack_types)
            
            if attack_type == 'translate':
                dx = random.uniform(-50, 50)
                dy = random.uniform(-50, 50)
                gdf_attacked = self.apply_translate_attack(gdf_attacked, dx, dy)
            elif attack_type == 'rotate':
                angle = random.uniform(-180, 180)
                gdf_attacked = self.apply_rotate_attack(gdf_attacked, angle)
            elif attack_type == 'scale':
                scale_factor = random.uniform(0.5, 2.0)
                gdf_attacked = self.apply_scale_attack(gdf_attacked, scale_factor)
            elif attack_type == 'noise':
                pct = random.choice([30, 50, 70])
                strength = random.uniform(0.4, 0.8)
                gdf_attacked = self.apply_noise_attack(gdf_attacked, pct, strength)
            elif attack_type == 'crop':
                crop_type = random.choice(['x_center_50pct', 'y_center_50pct', 'random_40pct', 'top_left'])
                gdf_attacked = self.apply_crop_attack(gdf_attacked, crop_type)
            elif attack_type == 'flip':
                flip_type = random.choice(['x', 'y', 'xy'])
                gdf_attacked = self.apply_flip_attack(gdf_attacked, flip_type)
            elif attack_type == 'add_vertices':
                pct = random.choice([20, 40, 60])
                noise_strength = random.choice([0, 1, 2])
                gdf_attacked = self.apply_add_vertices_attack(gdf_attacked, pct, noise_strength)
            elif attack_type == 'delete_vertices':
                pct = random.choice([10, 20, 30])
                gdf_attacked = self.apply_delete_vertices_attack(gdf_attacked, pct)
            
            # 防御性编程：每步后重置索引
            gdf_attacked = gdf_attacked.reset_index(drop=True)
        
        return gdf_attacked
    
    def apply_random_attack(self, gdf):
        """应用随机攻击（所有几何变换使用全局中心）"""
        attack_types = ['translate', 'rotate', 'scale', 'noise', 'crop', 'flip']
        attack_type = random.choice(attack_types)
        
        if attack_type == 'translate':
            dx = random.uniform(-30, 30)
            dy = random.uniform(-30, 30)
            return self.apply_translate_attack(gdf, dx, dy)
        elif attack_type == 'rotate':
            angle = random.uniform(-90, 90)
            return self.apply_rotate_attack(gdf, angle)
        elif attack_type == 'scale':
            scale_factor = random.uniform(0.7, 1.3)
            return self.apply_scale_attack(gdf, scale_factor)
        elif attack_type == 'noise':
            strength = random.uniform(0.1, 0.5)
            return self.apply_noise_attack(gdf, 50, strength)  # 改为50%而不是100%
        elif attack_type == 'crop':
            crop_type = random.choice(['x_center_50pct', 'y_center_50pct', 'random_40pct'])
            return self.apply_crop_attack(gdf, crop_type)
        elif attack_type == 'flip':
            flip_type = random.choice(['x', 'y', 'xy'])
            return self.apply_flip_attack(gdf, flip_type)
    
    def is_attack_affected(self, attack_name):
        """判断攻击是否受几何变换改动影响（局部中心→全局中心）"""
        # 受影响的单体攻击：所有涉及旋转、缩放、翻转的攻击
        affected_attacks = [
            "scale_0.5x", "scale_2x", "scale_x0.5_y2", "scale_x2_y0.5", "scale_random",
            "rotate_45", "rotate_90", "rotate_135", "rotate_180", "rotate_random",
            "flip_x", "flip_y", "flip_xy"
        ]
        
        # 检查是否为受影响的单体攻击
        for affected in affected_attacks:
            if affected in attack_name:
                return True
        
        # 所有组合攻击都视为受影响（因为可能包含rotate/scale/flip）
        if "combo_attack" in attack_name:
            return True
        
        # 扩展攻击（attack_051到attack_100）也可能使用随机攻击，视为受影响
        if attack_name.startswith("attack_") and attack_name.endswith(".geojson"):
            try:
                attack_num = int(attack_name.replace("attack_", "").replace(".geojson", ""))
                if attack_num > 50:  # attack_051及以后的扩展攻击
                    return True
            except ValueError:
                pass
        
        return False
    
    def clean_output_subdir(self, output_subdir):
        """清理输出子目录的旧文件"""
        subdir_path = os.path.join(self.output_dir, output_subdir)
        if os.path.exists(subdir_path):
            print(f"清理旧文件: {subdir_path}")
            shutil.rmtree(subdir_path)
        os.makedirs(subdir_path, exist_ok=True)
    
    def ensure_output_subdir(self, output_subdir):
        """确保输出子目录存在（不清空）"""
        subdir_path = os.path.join(self.output_dir, output_subdir)
        os.makedirs(subdir_path, exist_ok=True)

    def save_attacked_data(self, gdf, filename, attack_name, output_subdir):
        """保存被攻击的数据（增量更新模式），使用原子操作防止脏数据"""
        base_name = os.path.splitext(filename)[0]
        attack_base_name = os.path.splitext(attack_name)[0]
        output_filename = f"{attack_base_name}.geojson"
        output_path = os.path.join(self.output_dir, output_subdir, output_filename)
        
        # 检查是否需要重新生成
        if os.path.exists(output_path):
            if not self.is_attack_affected(attack_name):
                # 不受影响的攻击且文件已存在，跳过
                print(f"跳过（未受影响）: {output_filename}")
                return "skipped"
            else:
                # 受影响的攻击，需要覆盖
                print(f"覆盖（受影响）: {output_filename}")
        else:
            print(f"新建: {output_filename}")
        
        # 使用临时文件+重命名确保原子操作（防止脏数据）
        temp_path = output_path + '.tmp'
        try:
            # 先写入临时文件
            gdf.to_file(temp_path, driver='GeoJSON')
            
            # 验证文件有效性（尝试读取）
            test_gdf = gpd.read_file(temp_path)
            if len(test_gdf) == 0:
                raise ValueError("生成的文件为空")
            
            # 验证成功，原子性替换目标文件
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(temp_path, output_path)
            
            return True
        except Exception as e:
            print(f"  ❌ 保存失败: {e}")
            # 清理临时文件
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            return False
    
    def generate_attacks(self):
        """生成200个攻击版本（增量更新模式：仅重新生成受影响的文件）"""
        # 获取所有geojson文件
        geojson_files = [f for f in os.listdir(self.input_dir) if f.endswith('.geojson')]
        
        if not geojson_files:
            print("未找到geojson文件")
            return
        
        print(f"找到 {len(geojson_files)} 个矢量文件")
        print("\n【增量更新模式】")
        print("- 受影响的攻击（旋转/缩放/翻转相关）：强制重新生成")
        print("- 不受影响的攻击：保留已有文件，仅生成缺失文件")
        print("- 所有组合攻击：强制重新生成\n")
        
        for geojson_file in geojson_files:
            print(f"\n{'='*60}")
            print(f"处理文件: {geojson_file}")
            print(f"{'='*60}")
            gdf = self.load_vector_data(geojson_file)
            
            if gdf is None:
                continue
            
            # 获取文件名称（去掉.geojson后缀）作为子文件夹名
            file_base_name = os.path.splitext(geojson_file)[0]
            
            # 确保输出子目录存在（不清空）
            self.ensure_output_subdir(file_base_name)
            
            # 统计信息
            stats = {"generated": 0, "skipped": 0, "failed": 0}
            
            # 生成前100个指定攻击方式
            print(f"\n生成前100个指定攻击方式...")
            for i, (attack_name, attack_desc) in enumerate(tqdm(self.single_attacks, desc="指定攻击")):
                try:
                    gdf_attacked = self.apply_single_attack(gdf, attack_name)
                    
                    # 验证攻击结果有效性
                    if gdf_attacked is None or len(gdf_attacked) == 0:
                        print(f"  ⚠️  攻击 {attack_name} 产生空结果，跳过保存")
                        stats["failed"] += 1
                        continue
                    
                    # 不检查对象数，所有攻击结果都保存（图构建时会自适应K值）
                    result = self.save_attacked_data(gdf_attacked, geojson_file, attack_name, file_base_name)
                    if result == "skipped":
                        stats["skipped"] += 1
                    elif result:
                        stats["generated"] += 1
                    else:
                        stats["failed"] += 1
                except Exception as e:
                    print(f"应用攻击 {attack_name} 时出错: {e}")
                    stats["failed"] += 1
                    continue
            
            # 生成后100个随机组合攻击
            print(f"\n生成后100个随机组合攻击...")
            for i, (attack_name, attack_desc) in enumerate(tqdm(self.combo_attacks, desc="组合攻击")):
                try:
                    gdf_attacked = self.apply_combo_attack(gdf, attack_name)
                    
                    # 验证攻击结果有效性
                    if gdf_attacked is None or len(gdf_attacked) == 0:
                        print(f"  ⚠️  组合攻击 {attack_name} 产生空结果，跳过保存")
                        stats["failed"] += 1
                        continue
                    
                    # 不检查对象数，所有攻击结果都保存（图构建时会自适应K值）
                    result = self.save_attacked_data(gdf_attacked, geojson_file, attack_name, file_base_name)
                    if result == "skipped":
                        stats["skipped"] += 1
                    elif result:
                        stats["generated"] += 1
                    else:
                        stats["failed"] += 1
                except Exception as e:
                    print(f"应用组合攻击 {attack_name} 时出错: {e}")
                    stats["failed"] += 1
                    continue
            
            # 输出统计信息
            print(f"\n{'='*60}")
            print(f"文件 {geojson_file} 处理完成：")
            print(f"  ✅ 生成/覆盖: {stats['generated']} 个文件")
            print(f"  ⏭️  跳过: {stats['skipped']} 个文件")
            print(f"  ❌ 失败: {stats['failed']} 个文件")
            print(f"{'='*60}")

def main():
    """主函数"""
    print("=== 第一步：生成被攻击的矢量数据 ===")
    
    # 设置随机种子以确保可重复性
    random.seed(42)
    np.random.seed(42)
    
    # 创建攻击生成器
    generator = VectorAttackGenerator()
    
    # 生成攻击数据
    generator.generate_attacks()
    
    print("\n攻击数据生成完成！")
    print(f"攻击数据保存在: {generator.output_dir}")
    print("为每个图生成了200个被攻击的矢量地图数据")

if __name__ == "__main__":
    main() 