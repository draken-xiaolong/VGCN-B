#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第一步：生成测试集被攻击的矢量数据
按照test100.py的逻辑：前50个指定攻击方式，后50个随机组合攻击
为每个图生成100个被攻击的矢量地图类型放入vector_data_test的各个图的子文件夹中
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

class TestVectorAttackGenerator:
    """测试集矢量数据攻击生成器"""
    
    def __init__(self, input_dir="../convertToGeoJson/GeoJson/TestSet", output_dir="GeoJson-Attacked/TestSet"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.ensure_output_dir()
        
        # 定义前50个指定攻击方式
        self.single_attacks = [
            ("test_del_vertices_7pct.geojson", "随机删除 7% 顶点"),
            ("test_del_vertices_15pct.geojson", "随机删除 15% 顶点"),
            ("test_del_vertices_28pct.geojson", "随机删除 28% 顶点"),
            ("test_del_vertices_42pct.geojson", "随机删除 42% 顶点"),
            ("test_del_vertices_55pct.geojson", "随机删除 55% 顶点"),
            ("test_del_objects_8pct.geojson", "删除 8% 图形对象"),
            ("test_del_objects_18pct.geojson", "删除 18% 图形对象"),
            ("test_del_objects_33pct.geojson", "删除 33% 图形对象"),
            ("test_del_objects_46pct.geojson", "删除 46% 图形对象"),
            ("test_del_objects_60pct.geojson", "删除 60% 图形对象"),
            ("test_add_vertices_12pct.geojson", "添加 12% 顶点"),
            ("test_add_vertices_25pct.geojson", "添加 25% 顶点"),
            ("test_add_vertices_38pct.geojson", "添加 38% 顶点"),
            ("test_add_vertices_47pct.geojson", "添加 47% 顶点"),
            ("test_add_vertices_65pct.geojson", "添加 65% 顶点"),
            ("test_noise_vertices_8pct_0.25.geojson", "扰动 8% 顶点，强度 0.25"),
            ("test_noise_vertices_14pct_0.45.geojson", "扰动 14% 顶点，强度 0.45"),
            ("test_noise_vertices_21pct_0.55.geojson", "扰动 21% 顶点，强度 0.55"),
            ("test_noise_vertices_26pct_0.75.geojson", "扰动 26% 顶点，强度 0.75"),
            ("test_noise_vertices_33pct_0.85.geojson", "扰动 33% 顶点，强度 0.85"),
            ("test_crop_x_40pct.geojson", "沿 X 轴裁剪 40%"),
            ("test_crop_y_35pct.geojson", "沿 Y 轴裁剪 35%"),
            ("test_crop_top_25pct.geojson", "裁剪上部 25% 区域"),
            ("test_crop_bottom_20pct.geojson", "裁剪下部 20% 区域"),
            ("test_crop_random_30pct.geojson", "随机裁剪 30%"),
            ("test_translate_5_5.geojson", "平移 (5, 5)"),
            ("test_translate_15_-10.geojson", "平移 (15, -10)"),
            ("test_translate_-20_8.geojson", "平移 (-20, 8)"),
            ("test_translate_30_25.geojson", "平移 (30, 25)"),
            ("test_translate_-12_-12.geojson", "平移 (-12, -12)"),
            ("test_scale_0.65.geojson", "缩放 0.65 倍"),
            ("test_scale_1.5.geojson", "缩放 1.5 倍"),
            ("test_scale_x0.8_y1.4.geojson", "X 轴缩放 0.8，Y 轴缩放 1.4"),
            ("test_scale_x1.6_y0.7.geojson", "X 轴缩放 1.6，Y 轴缩放 0.7"),
            ("test_scale_random_0.4-2.5.geojson", "随机缩放（0.4–2.5）"),
            ("test_rotate_30.geojson", "旋转 30°"),
            ("test_rotate_75.geojson", "旋转 75°"),
            ("test_rotate_120.geojson", "旋转 120°"),
            ("test_rotate_225.geojson", "旋转 225°"),
            ("test_rotate_random.geojson", "随机旋转（0–360°）"),
            ("test_flip_x.geojson", "X 轴翻转"),
            ("test_flip_y.geojson", "Y 轴翻转"),
            ("test_flip_xy.geojson", "X、Y 轴同时翻转"),
            ("test_reverse_vertices.geojson", "反转顶点顺序"),
            ("test_reverse_objects.geojson", "反转对象顺序"),
            ("test_shuffle_objects.geojson", "打乱对象顺序"),
            ("test_shuffle_vertices.geojson", "打乱顶点顺序"),
            ("test_jitter_vertices_small.geojson", "小幅扰动顶点位置"),
            ("test_merge_objects_random.geojson", "随机合并对象"),
            ("test_split_objects_random.geojson", "随机拆分对象"),
        ]
        
        # 创建50个组合攻击
        self.combo_attacks = [(f"test_combo_attack_{i:03d}.geojson", f"组合攻击策略 {i}") for i in range(1, 51)]
    
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
    
    def apply_add_vertices_attack(self, gdf, percentage):
        """添加指定百分比的顶点"""
        def add_vertices_to_geom(geom, pct):
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
                        mid_point = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
                        new_coords.append(mid_point)
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
                        mid_point = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
                        new_ext_coords.append(mid_point)
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
                                mid_point = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
                                new_ring_coords.append(mid_point)
                        new_ring_coords.append(ring_coords[-1])
                        holes.append(new_ring_coords)
                    else:
                        holes.append(ring_coords)
                return Polygon(new_ext_coords, holes=holes if holes else None)
            return geom
        
        gdf_attacked = gdf.copy()
        gdf_attacked['geometry'] = gdf_attacked['geometry'].apply(
            lambda geom: add_vertices_to_geom(geom, percentage)
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
        
        if crop_type == "x_40pct":
            # 沿X轴裁剪40%
            mid_x = bounds[0] + (bounds[2] - bounds[0]) * 0.4
            gdf_attacked = gdf_attacked[bdf['minx'] < mid_x].reset_index(drop=True)
        elif crop_type == "y_35pct":
            # 沿Y轴裁剪35%
            mid_y = bounds[1] + (bounds[3] - bounds[1]) * 0.35
            gdf_attacked = gdf_attacked[bdf['miny'] < mid_y].reset_index(drop=True)
        elif crop_type == "top_25pct":
            # 裁剪上部25%区域
            top_y = bounds[3] - (bounds[3] - bounds[1]) * 0.25
            gdf_attacked = gdf_attacked[bdf['miny'] > top_y].reset_index(drop=True)
        elif crop_type == "bottom_20pct":
            # 裁剪下部20%区域
            bottom_y = bounds[1] + (bounds[3] - bounds[1]) * 0.2
            gdf_attacked = gdf_attacked[bdf['miny'] < bottom_y].reset_index(drop=True)
        elif crop_type == "random_30pct":
            # 随机裁剪30%
            num_objects = len(gdf_attacked)
            num_to_keep = int(num_objects * 0.7)
            if num_to_keep > 0:
                indices_to_keep = random.sample(range(num_objects), num_to_keep)
                gdf_attacked = gdf_attacked.iloc[indices_to_keep].reset_index(drop=True)
        
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
    
    def apply_shuffle_attack(self, gdf, shuffle_type):
        """打乱攻击"""
        gdf_attacked = gdf.copy()
        if shuffle_type == "objects":
            gdf_attacked = gdf_attacked.sample(frac=1).reset_index(drop=True)
        elif shuffle_type == "vertices":
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
        """应用单体攻击"""
        if "test_del_vertices_7pct" in attack_name:
            return self.apply_delete_vertices_attack(gdf, 7)
        elif "test_del_vertices_15pct" in attack_name:
            return self.apply_delete_vertices_attack(gdf, 15)
        elif "test_del_vertices_28pct" in attack_name:
            return self.apply_delete_vertices_attack(gdf, 28)
        elif "test_del_vertices_42pct" in attack_name:
            return self.apply_delete_vertices_attack(gdf, 42)
        elif "test_del_vertices_55pct" in attack_name:
            return self.apply_delete_vertices_attack(gdf, 55)
        elif "test_del_objects_8pct" in attack_name:
            return self.apply_delete_objects_attack(gdf, 8)
        elif "test_del_objects_18pct" in attack_name:
            return self.apply_delete_objects_attack(gdf, 18)
        elif "test_del_objects_33pct" in attack_name:
            return self.apply_delete_objects_attack(gdf, 33)
        elif "test_del_objects_46pct" in attack_name:
            return self.apply_delete_objects_attack(gdf, 46)
        elif "test_del_objects_60pct" in attack_name:
            return self.apply_delete_objects_attack(gdf, 60)
        elif "test_add_vertices_12pct" in attack_name:
            return self.apply_add_vertices_attack(gdf, 12)
        elif "test_add_vertices_25pct" in attack_name:
            return self.apply_add_vertices_attack(gdf, 25)
        elif "test_add_vertices_38pct" in attack_name:
            return self.apply_add_vertices_attack(gdf, 38)
        elif "test_add_vertices_47pct" in attack_name:
            return self.apply_add_vertices_attack(gdf, 47)
        elif "test_add_vertices_65pct" in attack_name:
            return self.apply_add_vertices_attack(gdf, 65)
        elif "test_noise_vertices_8pct_0.25" in attack_name:
            return self.apply_noise_attack(gdf, 8, 0.25)
        elif "test_noise_vertices_14pct_0.45" in attack_name:
            return self.apply_noise_attack(gdf, 14, 0.45)
        elif "test_noise_vertices_21pct_0.55" in attack_name:
            return self.apply_noise_attack(gdf, 21, 0.55)
        elif "test_noise_vertices_26pct_0.75" in attack_name:
            return self.apply_noise_attack(gdf, 26, 0.75)
        elif "test_noise_vertices_33pct_0.85" in attack_name:
            return self.apply_noise_attack(gdf, 33, 0.85)
        elif "test_crop_x_40pct" in attack_name:
            return self.apply_crop_attack(gdf, "x_40pct")
        elif "test_crop_y_35pct" in attack_name:
            return self.apply_crop_attack(gdf, "y_35pct")
        elif "test_crop_top_25pct" in attack_name:
            return self.apply_crop_attack(gdf, "top_25pct")
        elif "test_crop_bottom_20pct" in attack_name:
            return self.apply_crop_attack(gdf, "bottom_20pct")
        elif "test_crop_random_30pct" in attack_name:
            return self.apply_crop_attack(gdf, "random_30pct")
        elif "test_translate_5_5" in attack_name:
            return self.apply_translate_attack(gdf, 5, 5)
        elif "test_translate_15_-10" in attack_name:
            return self.apply_translate_attack(gdf, 15, -10)
        elif "test_translate_-20_8" in attack_name:
            return self.apply_translate_attack(gdf, -20, 8)
        elif "test_translate_30_25" in attack_name:
            return self.apply_translate_attack(gdf, 30, 25)
        elif "test_translate_-12_-12" in attack_name:
            return self.apply_translate_attack(gdf, -12, -12)
        elif "test_scale_0.65" in attack_name:
            return self.apply_scale_attack(gdf, 0.65)
        elif "test_scale_1.5" in attack_name:
            return self.apply_scale_attack(gdf, 1.5)
        elif "test_scale_x0.8_y1.4" in attack_name:
            return self.apply_scale_attack(gdf, 0.8, 1.4)
        elif "test_scale_x1.6_y0.7" in attack_name:
            return self.apply_scale_attack(gdf, 1.6, 0.7)
        elif "test_scale_random_0.4-2.5" in attack_name:
            return self.apply_scale_attack(gdf, random.uniform(0.4, 2.5))
        elif "test_rotate_30" in attack_name:
            return self.apply_rotate_attack(gdf, 30)
        elif "test_rotate_75" in attack_name:
            return self.apply_rotate_attack(gdf, 75)
        elif "test_rotate_120" in attack_name:
            return self.apply_rotate_attack(gdf, 120)
        elif "test_rotate_225" in attack_name:
            return self.apply_rotate_attack(gdf, 225)
        elif "test_rotate_random" in attack_name:
            return self.apply_rotate_attack(gdf, random.uniform(0, 360))
        elif "test_flip_x" in attack_name:
            return self.apply_flip_attack(gdf, "x")
        elif "test_flip_y" in attack_name:
            return self.apply_flip_attack(gdf, "y")
        elif "test_flip_xy" in attack_name:
            return self.apply_flip_attack(gdf, "xy")
        elif "test_reverse_vertices" in attack_name:
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
        elif "test_reverse_objects" in attack_name:
            return gdf.iloc[::-1].reset_index(drop=True)
        elif "test_shuffle_objects" in attack_name:
            return self.apply_shuffle_attack(gdf, "objects")
        elif "test_shuffle_vertices" in attack_name:
            return self.apply_shuffle_attack(gdf, "vertices")
        elif "test_jitter_vertices_small" in attack_name:
            return self.apply_noise_attack(gdf, 15, 0.05)
        elif "test_merge_objects_random" in attack_name:
            return self.apply_merge_objects_attack(gdf)
        elif "test_split_objects_random" in attack_name:
            return self.apply_split_objects_attack(gdf)
        else:
            # 对于扩展攻击策略，使用随机攻击
            return self.apply_random_attack(gdf)
    
    def apply_combo_attack(self, gdf, attack_name):
        """应用组合攻击（所有几何变换使用全局中心）"""
        gdf_attacked = gdf.copy()
        
        # 随机选择2-3种攻击方式组合
        num_attacks = random.randint(2, 3)
        attack_types = ['translate', 'rotate', 'scale', 'noise', 'crop', 'flip']
        
        for _ in range(num_attacks):
            attack_type = random.choice(attack_types)
            
            if attack_type == 'translate':
                dx = random.uniform(-30, 30)
                dy = random.uniform(-30, 30)
                gdf_attacked = self.apply_translate_attack(gdf_attacked, dx, dy)
            elif attack_type == 'rotate':
                angle = random.uniform(-90, 90)
                gdf_attacked = self.apply_rotate_attack(gdf_attacked, angle)
            elif attack_type == 'scale':
                scale_factor = random.uniform(0.7, 1.3)
                gdf_attacked = self.apply_scale_attack(gdf_attacked, scale_factor)
            elif attack_type == 'noise':
                strength = random.uniform(0.1, 0.5)
                gdf_attacked = self.apply_noise_attack(gdf_attacked, 50, strength)  # 改为50%而不是100%
            elif attack_type == 'crop':
                crop_type = random.choice(['x_40pct', 'y_35pct', 'random_30pct'])
                gdf_attacked = self.apply_crop_attack(gdf_attacked, crop_type)
            elif attack_type == 'flip':
                flip_type = random.choice(['x', 'y', 'xy'])
                gdf_attacked = self.apply_flip_attack(gdf_attacked, flip_type)
            
            # ✅ 防御性编程：每一步后都确保索引连续
            # 避免某些攻击改变数据结构后，后续步骤使用错误的索引
            gdf_attacked = gdf_attacked.reset_index(drop=True)
        
        return gdf_attacked
    
    def apply_random_attack(self, gdf):
        """应用随机攻击（所有几何变换使用全局中心）"""
        attack_types = ['translate', 'rotate', 'scale', 'noise', 'crop', 'flip']
        attack_type = random.choice(attack_types)
        
        if attack_type == 'translate':
            dx = random.uniform(-20, 20)
            dy = random.uniform(-20, 20)
            return self.apply_translate_attack(gdf, dx, dy)
        elif attack_type == 'rotate':
            angle = random.uniform(-45, 45)
            return self.apply_rotate_attack(gdf, angle)
        elif attack_type == 'scale':
            scale_factor = random.uniform(0.8, 1.2)
            return self.apply_scale_attack(gdf, scale_factor)
        elif attack_type == 'noise':
            strength = random.uniform(0.05, 0.3)
            return self.apply_noise_attack(gdf, 50, strength)  # 改为50%而不是100%
        elif attack_type == 'crop':
            crop_type = random.choice(['x_40pct', 'y_35pct', 'random_30pct'])
            return self.apply_crop_attack(gdf, crop_type)
        elif attack_type == 'flip':
            flip_type = random.choice(['x', 'y', 'xy'])
            return self.apply_flip_attack(gdf, flip_type)
    
    def clean_output_subdir(self, output_subdir):
        """清理输出子目录的旧文件"""
        subdir_path = os.path.join(self.output_dir, output_subdir)
        if os.path.exists(subdir_path):
            print(f"清理旧文件: {subdir_path}")
            shutil.rmtree(subdir_path)
        os.makedirs(subdir_path, exist_ok=True)

    def save_attacked_data(self, gdf, filename, attack_name, output_subdir):
        """保存被攻击的数据，使用原子操作防止脏数据"""
        base_name = os.path.splitext(filename)[0]
        attack_base_name = os.path.splitext(attack_name)[0]
        output_filename = f"{attack_base_name}.geojson"
        output_path = os.path.join(self.output_dir, output_subdir, output_filename)
        
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
            
            print(f"保存攻击数据: {output_filename}")
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
        """生成100个攻击版本（前50个指定方式，后50个随机组合）"""
        # 获取所有geojson文件
        geojson_files = [f for f in os.listdir(self.input_dir) if f.endswith('.geojson')]
        
        if not geojson_files:
            print("未找到geojson文件")
            return
        
        print(f"找到 {len(geojson_files)} 个矢量文件")
        
        for geojson_file in geojson_files:
            print(f"\n处理文件: {geojson_file}")
            gdf = self.load_vector_data(geojson_file)
            
            if gdf is None:
                continue
            
            # 获取文件名称（去掉.geojson后缀）作为子文件夹名
            file_base_name = os.path.splitext(geojson_file)[0]
            
            # 清理并创建输出子目录
            self.clean_output_subdir(file_base_name)
            
            # 生成前50个指定攻击方式
            print(f"生成前50个指定攻击方式...")
            for i, (attack_name, attack_desc) in enumerate(tqdm(self.single_attacks, desc="指定攻击")):
                try:
                    gdf_attacked = self.apply_single_attack(gdf, attack_name)
                    
                    # 验证攻击结果有效性
                    if gdf_attacked is None or len(gdf_attacked) == 0:
                        print(f"  ⚠️  攻击 {attack_name} 产生空结果，跳过保存")
                        continue
                    
                    # 不检查对象数，所有攻击结果都保存（图构建时会自适应K值）
                    self.save_attacked_data(gdf_attacked, geojson_file, attack_name, file_base_name)
                except Exception as e:
                    print(f"  ❌ 应用攻击 {attack_name} 时出错: {e}")
                    continue
            
            # 生成后50个随机组合攻击
            print(f"生成后50个随机组合攻击...")
            for i, (attack_name, attack_desc) in enumerate(tqdm(self.combo_attacks, desc="组合攻击")):
                try:
                    gdf_attacked = self.apply_combo_attack(gdf, attack_name)
                    
                    # 验证攻击结果有效性
                    if gdf_attacked is None or len(gdf_attacked) == 0:
                        print(f"  ⚠️  组合攻击 {attack_name} 产生空结果，跳过保存")
                        continue
                    
                    # 不检查对象数，所有攻击结果都保存（图构建时会自适应K值）
                    self.save_attacked_data(gdf_attacked, geojson_file, attack_name, file_base_name)
                except Exception as e:
                    print(f"  ❌ 应用组合攻击 {attack_name} 时出错: {e}")
                    continue

def main():
    """主函数"""
    print("=== 第一步：生成测试集被攻击的矢量数据 ===")
    
    # 设置随机种子以确保可重复性
    random.seed(42)
    np.random.seed(42)
    
    # 创建攻击生成器
    generator = TestVectorAttackGenerator()
    
    # 生成攻击数据
    generator.generate_attacks()
    
    print("\n测试集攻击数据生成完成！")
    print(f"攻击数据保存在: {generator.output_dir}")
    print("为每个图生成了100个被攻击的矢量地图数据")

if __name__ == "__main__":
    main()
