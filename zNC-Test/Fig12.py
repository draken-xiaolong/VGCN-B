#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig12：复合攻击（一次性按 Fig1→Fig10 顺序依次执行），对每张矢量地图顺序执行：
1) 顶点删除 10%
2) 顶点增加 强度1 比例50%
3) 对象删除 50%
4) 噪声扰动 强度0.8 比例50%
5) 沿Y轴中心裁剪50%
6) 平移 X=20, Y=40
7) 缩放 90%
8) 顺时针旋转 180°
9) Y轴镜像翻转
10) 顺序操作：反转顶点顺序 → 反转对象顺序

输出目录隔离在 `vector-data-geojson-attacked/compound_seq` 与 `vector-data-geojson-attacked-graph/compound_seq`；
结果输出在 `zNC-Test/NC-Results/Fig12`。
"""

from pathlib import Path
import os
import sys
from typing import List, Dict
import pickle
import shutil
import random

import numpy as np

# 控制台 UTF-8
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# 强制刷新输出（确保实时写入日志）
import builtins
_original_print = builtins.print

def flush_print(*args, **kwargs):
    """带自动刷新的print函数"""
    _original_print(*args, **kwargs)
    sys.stdout.flush()

# 将print替换为flush_print以确保实时输出
builtins.print = flush_print

# 导入共享模块
try:
    from fig_common import (
        PROJECT_ROOT, MODEL_PATH, CAT32_PATH, K_FOR_KNN,
        extract_features_20d, gdf_to_graph, load_improved_gat_model,
        load_cat32, features_to_matrix, calc_nc, extract_features_from_graph,
        convert_to_geojson, convert_geojsons_to_graphs,
        generate_scramble_key, scramble_image, descramble_image,
        save_scramble_info, load_scramble_info
    )
except ImportError as e:
    print(f"无法导入 fig_common 模块: {e}")
    print("请确保 fig_common.py 在同一目录下")
    sys.exit(1)

try:
    import geopandas as gpd  # type: ignore
except Exception as exc:
    print("需要安装 geopandas: pip install geopandas fiona pyproj shapely")
    print("geopandas_import_error", exc)
    gpd = None  # type: ignore

try:
    import torch  # type: ignore
    from torch_geometric.data import Data  # type: ignore
except Exception as exc:
    print("需要安装 torch 和 torch-geometric")
    print("torch_import_error", exc)
    Data = None  # type: ignore

try:
    from sklearn.preprocessing import StandardScaler  # type: ignore
    from sklearn.neighbors import NearestNeighbors, kneighbors_graph  # type: ignore
except Exception as exc:
    print("需要安装 scikit-learn: pip install scikit-learn")
    print("sklearn_import_error", exc)
    StandardScaler = None  # type: ignore
    NearestNeighbors = None  # type: ignore
    kneighbors_graph = None  # type: ignore

try:
    import pandas as pd  # type: ignore
    # ⚠️ 关键：必须在任何matplotlib操作之前设置后端
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端

    # 配置中文字体（在设置后端之后）
    matplotlib.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "Microsoft JhengHei",
        "WenQuanYi Zen Hei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False

    import matplotlib.pyplot as plt  # type: ignore
    from PIL import Image  # type: ignore
except Exception as exc:
    print("需要安装 pandas, matplotlib, pillow")
    print("pandas_import_error", exc)
    pd = None  # type: ignore
    plt = None  # type: ignore
    Image = None  # type: ignore

# 目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
GRAPH_SUFFIX = os.environ.get('VGAT_GRAPH_SUFFIX', '').strip()
RESULTS_SUFFIX = os.environ.get('VGAT_RESULTS_SUFFIX', '').strip()

DIR_VECTOR = SCRIPT_DIR / 'vector-data'
DIR_VECTOR_GEOJSON = SCRIPT_DIR / 'vector-data-geojson'
DIR_VECTOR_GEOJSON_ATTACKED = SCRIPT_DIR / 'vector-data-geojson-attacked' / 'compound_seq'
DIR_GRAPH = SCRIPT_DIR / f'vector-data-geojson-attacked-graph{GRAPH_SUFFIX}'
DIR_GRAPH_ORIGINAL = DIR_GRAPH / 'Original'
DIR_GRAPH_ATTACKED = DIR_GRAPH / 'Attacked' / 'compound_seq'
DIR_ZEROWM = SCRIPT_DIR / f'vector-data-zerowatermark{RESULTS_SUFFIX}'
DIR_RESULTS = SCRIPT_DIR / 'NC-Results' / f'Fig12{RESULTS_SUFFIX}'

def step1_discover_inputs() -> List[Path]:
    print('[Step1] 扫描输入数据: ', DIR_VECTOR)
    if not DIR_VECTOR.exists():
        print('未找到目录: ', DIR_VECTOR)
        return []
    files: List[Path] = []
    files.extend(sorted(DIR_VECTOR.glob('*.shp')))
    files.extend(sorted(DIR_VECTOR.glob('*.geojson')))
    selected = files[:8]  # 选择前8个文件
    print('发现文件: ', [p.name for p in selected])
    return selected

def step2_convert_to_geojson(inputs: List[Path]) -> List[Path]:
    """按 convertToGeoJson 逻辑转为 GeoJSON。"""
    print('[Step2] 转换为 GeoJSON ->', DIR_VECTOR_GEOJSON)
    return convert_to_geojson(inputs, DIR_VECTOR_GEOJSON)

def step3_generate_compound_seq_attacks(original_geojsons: List[Path]) -> Dict[str, Dict[str, Path]]:
    """对每个 base，按 Fig1→Fig10 顺序一次性依次执行并仅输出一个最终攻击版本。"""
    print('[Step3] 生成复合(顺序)攻击 ->', DIR_VECTOR_GEOJSON_ATTACKED)
    if gpd is None:
        print('缺少 geopandas，无法生成攻击。')
        return {}

    random.seed(42)
    try:
        np.random.seed(42)
    except Exception:
        pass

    from shapely.geometry import LineString, Polygon  # type: ignore
    from shapely.affinity import translate as shp_translate, scale as shp_scale, rotate as shp_rotate  # type: ignore

    def delete_vertices_from_geom(geom, pct):
        try:
            if geom.geom_type == 'LineString':
                coords = list(geom.coords)
                if len(coords) <= 2:
                    return geom
                n_to_delete = max(1, int((len(coords) - 2) * pct / 100))
                if n_to_delete >= len(coords) - 2:
                    return geom
                idx = list(range(1, len(coords) - 1))
                to_del = set(random.sample(idx, min(n_to_delete, len(idx))))
                new_coords = [coords[0]] + [coords[i] for i in idx if i not in to_del] + [coords[-1]]
                return LineString(new_coords)
            elif geom.geom_type == 'Polygon':
                ext = list(geom.exterior.coords)
                if len(ext) <= 4:
                    return geom
                n_to_delete = max(1, int((len(ext) - 4) * pct / 100))
                if n_to_delete >= len(ext) - 4:
                    return geom
                idx = list(range(1, len(ext) - 2))
                to_del = set(random.sample(idx, min(n_to_delete, len(idx))))
                new_ext = [ext[0]] + [ext[i] for i in range(1, len(ext) - 2) if i not in to_del] + [ext[-2], ext[-1]]
                holes = []
                for ring in geom.interiors:
                    rc = list(ring.coords)
                    if len(rc) > 4:
                        n_h = max(1, int((len(rc) - 4) * pct / 100))
                        if n_h < len(rc) - 4:
                            idx_h = list(range(1, len(rc) - 2))
                            del_h = set(random.sample(idx_h, min(n_h, len(idx_h))))
                            holes.append([rc[0]] + [rc[i] for i in range(1, len(rc) - 2) if i not in del_h] + [rc[-2], rc[-1]])
                        else:
                            holes.append(rc)
                    else:
                        holes.append(rc)
                return Polygon(new_ext, holes=holes if holes else None)
        except Exception:
            pass
        return geom

    def add_vertices_to_geom(geom, pct, strength_level=1):
        noise_sigma = 0.01 if strength_level == 1 else 0.0
        try:
            if geom.geom_type == 'LineString':
                coords = list(geom.coords)
                if len(coords) < 2:
                    return geom
                n_to_add = min(3, max(1, int((len(coords) - 1) * pct / 100)))
                new_coords = [coords[0]]
                for i in range(len(coords) - 1):
                    p1, p2 = coords[i], coords[i + 1]
                    new_coords.append(p1)
                    for j in range(n_to_add):
                        t = (j + 1) / (n_to_add + 1)
                        mid = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
                        if noise_sigma > 0:
                            mid = (mid[0] + float(np.random.normal(0, noise_sigma)), mid[1] + float(np.random.normal(0, noise_sigma)))
                        new_coords.append(mid)
                new_coords.append(coords[-1])
                return LineString(new_coords)
            elif geom.geom_type == 'Polygon':
                ext = list(geom.exterior.coords)
                if len(ext) < 4:
                    return geom
                n_to_add = min(3, max(1, int((len(ext) - 1) * pct / 100)))
                new_ext = [ext[0]]
                for i in range(len(ext) - 1):
                    p1, p2 = ext[i], ext[i + 1]
                    new_ext.append(p1)
                    for j in range(n_to_add):
                        t = (j + 1) / (n_to_add + 1)
                        mid = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
                        if noise_sigma > 0:
                            mid = (mid[0] + float(np.random.normal(0, noise_sigma)), mid[1] + float(np.random.normal(0, noise_sigma)))
                        new_ext.append(mid)
                new_ext.append(ext[-1])
                holes = []
                for ring in geom.interiors:
                    rc = list(ring.coords)
                    if len(rc) >= 4:
                        new_rc = [rc[0]]
                        for i in range(len(rc) - 1):
                            p1, p2 = rc[i], rc[i + 1]
                            new_rc.append(p1)
                            for j in range(n_to_add):
                                t = (j + 1) / (n_to_add + 1)
                                mid = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
                                if noise_sigma > 0:
                                    mid = (mid[0] + float(np.random.normal(0, noise_sigma)), mid[1] + float(np.random.normal(0, noise_sigma)))
                                new_rc.append(mid)
                        new_rc.append(rc[-1])
                        holes.append(new_rc)
                    else:
                        holes.append(rc)
                return Polygon(new_ext, holes=holes if holes else None)
        except Exception:
            pass
        return geom

    def jitter_vertices(geom, pct, strength):
        try:
            if geom.geom_type == 'LineString':
                coords = list(geom.coords)
                n = len(coords)
                k = max(1, int(n * pct / 100))
                idx = list(range(n))
                chosen = set(random.sample(idx, min(k, len(idx))))
                new_coords = []
                for i, c in enumerate(coords):
                    if i in chosen:
                        new_coords.append((c[0] + random.uniform(-strength, strength), c[1] + random.uniform(-strength, strength)))
                    else:
                        new_coords.append(c)
                return LineString(new_coords)
            elif geom.geom_type == 'Polygon':
                ext = list(geom.exterior.coords)
                n = len(ext)
                k = max(1, int(n * pct / 100))
                idx = list(range(n))
                chosen = set(random.sample(idx, min(k, len(idx))))
                new_ext = []
                for i, c in enumerate(ext):
                    if i in chosen:
                        new_ext.append((c[0] + random.uniform(-strength, strength), c[1] + random.uniform(-strength, strength)))
                    else:
                        new_ext.append(c)
                holes = []
                for ring in geom.interiors:
                    rc = list(ring.coords)
                    n2 = len(rc); k2 = max(1, int(n2 * pct / 100))
                    idx2 = list(range(n2)); chosen2 = set(random.sample(idx2, min(k2, len(idx2))))
                    new_rc = []
                    for i, c in enumerate(rc):
                        if i in chosen2:
                            new_rc.append((c[0] + random.uniform(-strength, strength), c[1] + random.uniform(-strength, strength)))
                        else:
                            new_rc.append(c)
                    holes.append(new_rc)
                return Polygon(new_ext, holes=holes if holes else None)
        except Exception:
            pass
        return geom

    def reverse_vertices_geom(geom):
        try:
            if geom.geom_type == 'LineString':
                return LineString(list(geom.coords)[::-1])
            elif geom.geom_type == 'Polygon':
                ext = list(geom.exterior.coords)
                ext = ext[:-1][::-1] + [ext[0]]
                holes = []
                for ring in geom.interiors:
                    rc = list(ring.coords)
                    rc = rc[:-1][::-1] + [rc[0]]
                    holes.append(rc)
                return Polygon(ext, holes=holes if holes else None)
        except Exception:
            pass
        return geom

    outputs: Dict[str, Dict[str, Path]] = {}
    DIR_VECTOR_GEOJSON_ATTACKED.mkdir(parents=True, exist_ok=True)

    for src in original_geojsons:
        base = src.stem
        subdir = DIR_VECTOR_GEOJSON_ATTACKED / base
        if subdir.exists():
            shutil.rmtree(subdir)
        subdir.mkdir(parents=True, exist_ok=True)

        try:
            gdf = gpd.read_file(src)
        except Exception as exc:
            print('read_geojson_error', src.name, exc)
            continue

        # 计算裁剪辅助
        bounds = gdf.total_bounds
        mid_y = (bounds[1] + bounds[3]) / 2

        # 1 顶点删除10%
        gdf['geometry'] = gdf['geometry'].apply(lambda geom: delete_vertices_from_geom(geom, 10))
        # 2 顶点增加 强度1 比例50%
        gdf['geometry'] = gdf['geometry'].apply(lambda geom: add_vertices_to_geom(geom, 50, strength_level=1))
        # 3 对象删除50%
        n_total = len(gdf)
        n_del = int(n_total * 0.5)
        if n_del > 0 and n_total > 0:
            idx = random.sample(range(n_total), min(n_del, n_total))
            gdf = gdf.drop(idx).reset_index(drop=True)
        # 4 噪声扰动 强度0.8 比例50%
        gdf['geometry'] = gdf['geometry'].apply(lambda geom: jitter_vertices(geom, 50, 0.8))
        # 5 沿Y轴中心裁剪50%
        bdf = gdf.geometry.bounds
        gdf = gdf[bdf['miny'] < mid_y].reset_index(drop=True)
        # 6 平移 X=20, Y=40
        gdf['geometry'] = gdf['geometry'].apply(lambda geom: shp_translate(geom, 20, 40))

        # 计算当前全局中心（用于后续的缩放、旋转、翻转）
        bounds_current = gdf.total_bounds
        global_center = ((bounds_current[0] + bounds_current[2]) / 2, (bounds_current[1] + bounds_current[3]) / 2)

        # 7 缩放90%（使用全局中心）
        gdf['geometry'] = gdf['geometry'].apply(lambda geom: shp_scale(geom, 0.9, 0.9, origin=global_center))
        # 8 旋转180°（使用全局中心）
        gdf['geometry'] = gdf['geometry'].apply(lambda geom: shp_rotate(geom, 180, origin=global_center))
        # 9 Y轴镜像翻转（使用全局中心）
        gdf['geometry'] = gdf['geometry'].apply(lambda geom: shp_scale(geom, 1.0, -1.0, origin=global_center))
        # 10 反转顶点顺序 → 反转对象顺序
        gdf['geometry'] = gdf['geometry'].apply(reverse_vertices_geom)
        gdf = gdf.iloc[::-1].reset_index(drop=True)

        out_path = subdir / 'compound_seq_all.geojson'
        try:
            gdf.to_file(out_path, driver='GeoJSON')
            print(f'输出: {base}/{out_path.name} ({len(gdf)} 要素)')
        except Exception as exc:
            print('attack_save_error', base, exc)
            continue

        outputs.setdefault(base, {})['compound_seq_all'] = out_path

    return outputs

def step4_convert_to_graph(original_geojsons: List[Path], attacked_map: Dict[str, Dict[str, Path]]):
    """使用 fig_common 的标准函数转为图结构（KNN无向图 k=8，13维特征，与训练集一致）。"""
    print('[Step4] 转换为图结构 ->', DIR_GRAPH)

    if gpd is None or Data is None:
        print('缺少依赖，无法构图。')
        return

    # 使用共享函数批量转换（13维特征 + KNN无向图，与训练集一致）
    convert_geojsons_to_graphs(
        original_geojsons=original_geojsons,
        attacked_geojson_map=attacked_map,
        output_dir_original=DIR_GRAPH_ORIGINAL,
        output_dir_attacked=DIR_GRAPH_ATTACKED
    )

def step5_generate_zero_watermark():
    """用 VGAT 模型对 Original 生成零水印（带置乱）。"""
    print('[Step5] 生成零水印（带置乱） ->', DIR_ZEROWM)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, device = load_improved_gat_model(device)
    copyright_img = load_cat32()

    DIR_ZEROWM.mkdir(parents=True, exist_ok=True)

    cnt = 0
    for pkl in sorted(DIR_GRAPH_ORIGINAL.glob('*_graph.pkl')):
        try:
            with open(pkl, 'rb') as f:
                graph = pickle.load(f)

            base = pkl.stem.replace('_graph', '')
            
            # ⭐生成置乱密钥
            scramble_key, timestamp = generate_scramble_key(base)
            print(f'生成置乱密钥 - 数据集: {base}, 密钥: {scramble_key}')
            
            # ⭐置乱版权图像
            scrambled_copyright = scramble_image(copyright_img, scramble_key)
            
            feat_mat = extract_features_from_graph(graph, model, device, copyright_img.shape)
            
            # ⭐使用置乱后的版权图像生成零水印
            zwm = np.logical_xor(feat_mat, scrambled_copyright).astype(np.uint8)

            np.save(DIR_ZEROWM / f'{base}_watermark.npy', zwm)
            try:
                from PIL import Image
                Image.fromarray((zwm * 255).astype(np.uint8)).save(DIR_ZEROWM / f'{base}_watermark.png')
            except Exception:
                pass
            
            # ⭐保存置乱信息
            save_scramble_info(base, scramble_key, timestamp, DIR_ZEROWM / f'{base}_scramble_info.json')
            
            print(f'零水印已保存: {base}_watermark.npy (置乱密钥: {scramble_key})')
            cnt += 1
        except Exception as exc:
            print('zero_watermark_error', pkl.name, exc)
            continue

    if cnt == 0:
        print('未找到 Original 图，无法生成零水印。')
    else:
        print(f'共生成零水印 {cnt} 个（已应用置乱）。')

def step6_evaluate_nc():
    print('[Step6] 评估NC（带置乱恢复） ->', DIR_RESULTS)
    if not DIR_GRAPH_ATTACKED.exists():
        print('缺少攻击图目录: ', DIR_GRAPH_ATTACKED); print('请先运行 Step4 转换为图结构。'); return
    if Data is None or not MODEL_PATH.exists():
        if not MODEL_PATH.exists(): print('模型文件不存在: ', MODEL_PATH)
        else: print('缺少 torch/torch-geometric 依赖。')
        return
    try:
        import pandas as pd  # type: ignore
        # 设置matplotlib使用非交互式后端（必须在导入pyplot之前）

        import matplotlib
        matplotlib.use('Agg')

        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        print('需要安装 pandas/matplotlib'); print(exc); return

    # 使用共享函数加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model, device = load_improved_gat_model(device)
    except Exception as exc:
        print(f'模型加载失败: {exc}')
        return

    DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    # 创建提取水印图的输出目录
    extract_dir = SCRIPT_DIR / 'extract' / 'watermark' / 'Fig12'
    extract_dir.mkdir(parents=True, exist_ok=True)

    copyright_img = load_cat32()

    rows = []
    for base_dir_path in sorted(DIR_GRAPH_ATTACKED.iterdir()):
        if not base_dir_path.is_dir(): continue
        base = base_dir_path.name
        zwm_path = DIR_ZEROWM / f'{base}_watermark.npy'
        scramble_info_path = DIR_ZEROWM / f'{base}_scramble_info.json'
        
        if not zwm_path.exists(): 
            print('缺少零水印，跳过: ', base)
            continue
        
        # ⭐加载置乱信息
        if not scramble_info_path.exists():
            print(f'警告: 缺少置乱信息文件，跳过: {base}')
            continue
        
        _, scramble_key, _ = load_scramble_info(scramble_info_path)
        zwm = np.load(zwm_path)
        
        pkl = base_dir_path / 'compound_seq_all_graph.pkl'
        if not pkl.exists():
            print('缺少复合攻击图: ', pkl); continue
        try:
            with open(pkl,'rb') as f: graph: Data = pickle.load(f)
            feat_mat = extract_features_from_graph(graph, model, device, copyright_img.shape)
            
            # ⭐从零水印提取置乱后的版权图像
            scrambled_extracted = np.logical_xor(zwm, feat_mat).astype(np.uint8)
            
            # ⭐使用置乱密钥恢复版权图像
            extracted = descramble_image(scrambled_extracted, scramble_key)
            
            # ⭐计算NC值
            nc = calc_nc(copyright_img, extracted)
            rows.append({'base': base, 'nc': nc})

            # 保存提取的水印图片 (参考对比实验的做法)
            try:
                out_img_path = extract_dir / f'Cat32_{base}_extracted.png'
                extracted_img = (extracted * 255).astype(np.uint8)
                Image.fromarray(extracted_img).save(out_img_path)
                print(f'已保存提取水印图: {out_img_path.name}')
            except Exception as img_exc:
                print(f'保存提取水印图失败 {base}: {img_exc}')
        except Exception as exc:
            print('nc_eval_error', pkl.name, exc)

    if not rows:
        print('没有评估结果。'); return

    df = pd.DataFrame(rows).sort_values('base')

    # 层次表：仅一个"复合攻击(顺序)"分组
    hierarchical_rows = []
    hierarchical_rows.append({'复合攻击(顺序)': '复合(Fig1→Fig10)', 'VGAT': '', '类型': 'header'})
    for _, row in df.iterrows():
        hierarchical_rows.append({'复合攻击(顺序)': f"  {row['base']}", 'VGAT': f"{row['nc']:.6f}", '类型': 'data'})
    hierarchical_rows.append({'复合攻击(顺序)': '  Average', 'VGAT': f"{df['nc'].mean():.6f}", '类型': 'average'})

    hierarchical_df = pd.DataFrame(hierarchical_rows)
    csv_path = DIR_RESULTS / 'fig12_compound_seq_nc.csv'
    hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    xlsx_path = DIR_RESULTS / 'fig12_compound_seq_nc.xlsx'
    try:
        hierarchical_df.to_excel(xlsx_path, index=False, engine='openpyxl')
        print('结果表保存: ', csv_path.name)
        print('Excel文件保存: ', xlsx_path.name)
    except Exception as e:
        print('Excel保存失败: ', e)
        hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print('结果表保存: ', csv_path.name)

    # 条形图：各基底的NC + 平均
    plt.figure(figsize=(10, 6))
    bases = list(df['base'])
    vals = list(df['nc'])
    x = np.arange(len(bases))
    plt.bar(x, vals, alpha=0.8)
    plt.axhline(df['nc'].mean(), color='k', linestyle='--', label='平均')
    plt.xticks(x, bases, rotation=20)
    plt.ylabel('NC')
    plt.title('Fig12：复合攻击(一次性顺序 Fig1→Fig10) 的NC鲁棒性')
    plt.legend()
    fig_path = DIR_RESULTS / 'fig12_compound_seq_nc.png'
    plt.tight_layout(); plt.savefig(fig_path, dpi=300, bbox_inches='tight'); plt.close()
    print('柱状图保存: ', fig_path.name)

def main():
    print('=== Fig12：复合攻击(一次性顺序) 鲁棒性测试 ===')
    print('\n[从Step3开始] 跳过vector-data转换，直接使用vector-data-geojson...')
    # 直接使用现有的vector-data-geojson文件
    original_geojsons = [p for p in DIR_VECTOR_GEOJSON.glob('*.geojson') if not p.name.startswith('._')]
    if not original_geojsons:
        print('未找到vector-data-geojson文件')
        return
    print(f'发现 {len(original_geojsons)} 个GeoJSON文件: {[p.name for p in original_geojsons]}')
    attacked_map = step3_generate_compound_seq_attacks(original_geojsons)
    step4_convert_to_graph(original_geojsons, attacked_map)
    step5_generate_zero_watermark()
    step6_evaluate_nc()
    print('=== 完成 ===')

if __name__ == '__main__':
    main()
