#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig9：翻转攻击鲁棒性测试脚本（X轴镜像、Y轴镜像、同时XY镜像）

步骤与 Fig1 对齐，输出目录隔离在 `flip` 子目录。
"""

from pathlib import Path
import sys
from typing import List, Dict, Tuple
import pickle
import shutil
import random

import numpy as np

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
        convert_to_geojson, convert_geojsons_to_graphs
    ,
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
        "SimHei", "Microsoft YaHei", "Microsoft JhengHei",
        "WenQuanYi Zen Hei", "Noto Sans CJK SC", "DejaVu Sans",
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

DIR_VECTOR = SCRIPT_DIR / 'vector-data'
DIR_VECTOR_GEOJSON = SCRIPT_DIR / 'vector-data-geojson'
DIR_VECTOR_GEOJSON_ATTACKED = SCRIPT_DIR / 'vector-data-geojson-attacked' / 'flip'
DIR_GRAPH = SCRIPT_DIR / 'vector-data-geojson-attacked-graph'
DIR_GRAPH_ORIGINAL = DIR_GRAPH / 'Original'
DIR_GRAPH_ATTACKED = DIR_GRAPH / 'Attacked' / 'flip'
DIR_ZEROWM = SCRIPT_DIR / 'vector-data-zerowatermark'
DIR_RESULTS = SCRIPT_DIR / 'NC-Results' / 'Fig9'

# MODEL_PATH和CAT32_PATH已从fig_common导入，无需重复定义
# key, (sx, sy), label
# ⚠️ flip_xy必须放在前面，避免被flip_x/flip_y误匹配（"flip_x" in "flip_xy" => True）
FLIP_CONFIGS: List[Tuple[str, Tuple[float, float], str]] = [
    ("flip_xy", (-1.0, -1.0), "同时X、Y轴镜像翻转"),
    ("flip_x", (-1.0, 1.0), "X轴镜像翻转"),
    ("flip_y", (1.0, -1.0), "Y轴镜像翻转"),
]


def step1_discover_inputs() -> List[Path]:
    print('[Step1] : ', DIR_VECTOR)
    if not DIR_VECTOR.exists():
        print(' : ', DIR_VECTOR)
        return []
    files: List[Path] = []
    files.extend(sorted(DIR_VECTOR.glob('*.shp')))
    files.extend(sorted(DIR_VECTOR.glob('*.geojson')))
    selected = files[:8]
    print(' : ', [p.name for p in selected])
    return selected


def step2_convert_to_geojson(inputs: List[Path]) -> List[Path]:
    """ ."""
    print('[Step2] ->', DIR_VECTOR_GEOJSON)
    return convert_to_geojson(inputs, DIR_VECTOR_GEOJSON)
    for src in inputs:
        try:
            base = src.stem
            out_path = DIR_VECTOR_GEOJSON / f'{base}.geojson'
            gdf = gpd.read_file(src)
            if getattr(gdf, 'crs', None) and str(gdf.crs) != 'EPSG:4326':
                gdf = gdf.to_crs('EPSG:4326')
            gdf.to_file(out_path, driver='GeoJSON', encoding='utf-8')
            print(f' : {out_path.name} ({len(gdf)} )')
            outputs.append(out_path)
        except Exception as exc:
            print('convert_error', src.name, exc)
    return outputs


def step3_generate_flip_attacks(original_geojsons: List[Path]) -> Dict[str, Dict[str, Path]]:
    print('[Step3] ->', DIR_VECTOR_GEOJSON_ATTACKED)
    if gpd is None:
        print(' , .')
        return {}
    from shapely.affinity import scale as shp_scale, rotate as shp_rotate  # type: ignore
    DIR_VECTOR_GEOJSON_ATTACKED.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Dict[str, Path]] = {}
    for src in original_geojsons:
        base = src.stem
        subdir = DIR_VECTOR_GEOJSON_ATTACKED / base
        if subdir.exists(): shutil.rmtree(subdir)
        subdir.mkdir(parents=True, exist_ok=True)
        try:
            gdf = gpd.read_file(src)
        except Exception as exc:
            print('read_geojson_error', src.name, exc); continue
        
        # ✅ 使用全局中心作为变换原点，保持整个地图的相对位置关系
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        global_center_x = (bounds[0] + bounds[2]) / 2
        global_center_y = (bounds[1] + bounds[3]) / 2
        global_center = (global_center_x, global_center_y)
        
        outputs[base] = {}
        for key, (sx, sy), _label in FLIP_CONFIGS:
            attacked = gdf.copy()
            
            # ✅ 使用全局中心进行翻转和旋转，与Fig8策略一致
            if key == 'flip_xy':
                # 同时X、Y翻转 = 旋转180°（数学等价）
                # 使用rotate更稳定，避免scale(-1,-1)导致的顶点顺序反转问题
                attacked['geometry'] = attacked['geometry'].apply(
                    lambda geom: shp_rotate(geom, 180, origin=global_center)
                )
                print(f'  {base}: 双轴翻转使用旋转180°实现（全局中心，与Fig8一致）')
            else:
                # 单轴翻转使用scale，以全局中心为原点
                attacked['geometry'] = attacked['geometry'].apply(
                    lambda geom: shp_scale(geom, sx, sy, origin=global_center)
                )
            
            out_path = subdir / f'{key}.geojson'
            try:
                attacked.to_file(out_path, driver='GeoJSON')
                print(f' : {base}/{out_path.name}')
                outputs[base][key] = out_path
            except Exception as exc:
                print('attack_error', base, key, exc)
                continue
    return outputs


def step4_convert_to_graph(original_geojsons: List[Path], attacked_map: Dict[str, Dict[str, Path]]):
    """ ."""
    print('[Step4] ->', DIR_GRAPH)
    
    if gpd is None or Data is None:
        print(' , .')
        return
    
    # 
    convert_geojsons_to_graphs(
        original_geojsons=original_geojsons,
        attacked_geojson_map=attacked_map,
        output_dir_original=DIR_GRAPH_ORIGINAL,
        output_dir_attacked=DIR_GRAPH_ATTACKED
    )


def step5_generate_zero_watermark():
    """ ."""
    print('[Step5] ->', DIR_ZEROWM)
    
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
            
            # ⭐检查置乱信息文件是否已存在
            scramble_info_path = DIR_ZEROWM / f'{base}_scramble_info.json'
            if scramble_info_path.exists():
                # 使用现有的置乱密钥（固定不变）
                _, scramble_key, timestamp = load_scramble_info(scramble_info_path)
                print(f'使用现有置乱密钥 - 数据集: {base}, 密钥: {scramble_key}')
            else:
                # 首次生成置乱密钥
                scramble_key, timestamp = generate_scramble_key(base)
                print(f'生成新置乱密钥 - 数据集: {base}, 密钥: {scramble_key}')
            
            # ⭐置乱版权图像
            scrambled_copyright = scramble_image(copyright_img, scramble_key)
            
            # ⭐提取鲁棒特征并与置乱后的版权图像进行XOR生成零水印
            feat_mat = extract_features_from_graph(graph, model, device, copyright_img.shape)
            zwm = np.logical_xor(feat_mat, scrambled_copyright).astype(np.uint8)
            
            # ⭐保存零水印和置乱信息
            np.save(DIR_ZEROWM / f'{base}_watermark.npy', zwm)
            save_scramble_info(base, scramble_key, timestamp, DIR_ZEROWM / f'{base}_scramble_info.json')
            
            try:
                from PIL import Image
                Image.fromarray((zwm * 255).astype(np.uint8)).save(DIR_ZEROWM / f'{base}_watermark.png')
            except Exception:
                pass
            print(' : ', f'{base}_watermark.npy')
            cnt += 1
        except Exception as exc:
            print('zero_watermark_error', pkl.name, exc)
            continue
    
    if cnt == 0:
        print(' .')
    else:
        print(f' {cnt} .')


def step6_evaluate_nc():
    print('[Step6] ->', DIR_RESULTS)
    if not DIR_GRAPH_ATTACKED.exists():
        print(' : ', DIR_GRAPH_ATTACKED); print(' .'); return
    if Data is None or not MODEL_PATH.exists():
        if not MODEL_PATH.exists(): print(' : ', MODEL_PATH)
        else: print(' .')
        return
    # 检查依赖（pandas和plt已在文件开头导入）
    if pd is None or plt is None:
        print(' pandas/matplotlib'); return
    # 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model, device = load_improved_gat_model(device)
    except Exception as exc:
        print(f' : {exc}')
        return

    DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    copyright_img = load_cat32()

    key_to_label = {k: lbl for k, _, lbl in FLIP_CONFIGS}
    order_keys = [k for k, _, _ in FLIP_CONFIGS]
    rows = []
    for base_dir_path in sorted(DIR_GRAPH_ATTACKED.iterdir()):
        if not base_dir_path.is_dir(): continue
        base = base_dir_path.name
        zwm_path = DIR_ZEROWM / f'{base}_watermark.npy'
        scramble_info_path = DIR_ZEROWM / f'{base}_scramble_info.json'
        
        if not zwm_path.exists(): print(' , ', base); continue
        
        # ⭐加载置乱信息
        if not scramble_info_path.exists():
            print(f'警告: 缺少置乱信息文件，跳过: {base}')
            continue
        
        _, scramble_key, _ = load_scramble_info(scramble_info_path)
        zwm = np.load(zwm_path)
        for pkl in sorted(base_dir_path.glob('*_graph.pkl')):
            try:
                with open(pkl,'rb') as f: graph: Data = pickle.load(f)
                feat_mat = extract_features_from_graph(graph, model, device, copyright_img.shape)
                # ⭐从零水印提取置乱后的版权图像
                scrambled_extracted = np.logical_xor(zwm, feat_mat).astype(np.uint8)
                
                # ⭐使用置乱密钥恢复版权图像
                extracted = descramble_image(scrambled_extracted, scramble_key)
                
                # ⭐计算NC值
                nc = calc_nc(copyright_img, extracted)
                name = pkl.stem.replace('_graph','')
                k_found = None
                for k in order_keys:
                    if k in name: k_found = k; break
                rows.append({'base': base, 'key': k_found or 'unknown', 'nc': nc})
            except Exception as exc:
                print('nc_eval_error', pkl.name, exc)

    if not rows:
        print('.'); return

    df = pd.DataFrame(rows)
    hierarchical_rows = []
    base_names = sorted(df['base'].unique())
    for k in order_keys:
        label = key_to_label.get(k, k)
        hierarchical_rows.append({'翻转类型': label, 'VGAT': '', '类型': 'header'})
        for base in base_names:
            sub = df[(df['base']==base) & (df['key']==k)]
            nc_value = sub['nc'].iloc[0] if len(sub)>0 else 0
            hierarchical_rows.append({'翻转类型': f'  {base}', 'VGAT': f'{nc_value:.6f}', '类型': 'data'})
        avg_nc = df[df['key']==k]['nc'].mean()
        hierarchical_rows.append({'翻转类型': '  Average', 'VGAT': f'{avg_nc:.6f}', '类型': 'average'})

    hierarchical_df = pd.DataFrame(hierarchical_rows)
    csv_path = DIR_RESULTS / 'fig9_flip_nc.csv'
    hierarchical_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    xlsx_path = DIR_RESULTS / 'fig9_flip_nc.xlsx'
    try:
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            hierarchical_df.to_excel(writer, sheet_name='NC', index=False)
            ws = writer.sheets['NC']
            ws.column_dimensions['A'].width = 24
            ws.column_dimensions['B'].width = 15
            from openpyxl.styles import Font, Alignment
            for idx, row in hierarchical_df.iterrows():
                if row[''] == 'header':
                    ws[f'A{idx+2}'].font = Font(bold=True)
                elif row[''] == 'average':
                    ws[f'A{idx+2}'].font = Font(bold=True, italic=True)
                    ws[f'B{idx+2}'].font = Font(bold=True, italic=True)
                elif row[''] == 'data':
                    ws[f'A{idx+2}'].alignment = Alignment(horizontal='left', indent=1)
    except Exception as e:
        print('Excel : ', e)
        hierarchical_df.to_excel(xlsx_path, index=False, engine='openpyxl')
    print(' : ', csv_path.name)
    print('Excel : ', xlsx_path.name)

    # 
    plt.figure(figsize=(10,6))
    x = np.arange(len(FLIP_CONFIGS))
    labels = [lbl for _, _, lbl in FLIP_CONFIGS]
    for base, sub in df.groupby('base'):
        y = []
        for k, _, _ in FLIP_CONFIGS:
            val = sub[sub['key']==k]['nc']
            y.append(val.iloc[0] if len(val)>0 else np.nan)
        plt.plot(x, y, '-o', alpha=0.7, label=base)
    avg_vals = [df[df['key']==k]['nc'].mean() for k,_,_ in FLIP_CONFIGS]
    plt.plot(x, avg_vals, 'k-o', linewidth=2.5, label='')
    plt.grid(True, alpha=0.3)
    plt.xticks(x, labels)
    plt.xlabel('')
    plt.ylabel('NC')
    plt.title('Fig9： ')
    plt.legend(loc='best', fontsize=8, ncol=2)
    fig_path = DIR_RESULTS / 'fig9_flip_nc.png'
    plt.tight_layout(); plt.savefig(fig_path, dpi=300, bbox_inches='tight'); plt.close()
    print(' : ', fig_path.name)


def check_existing_files():
    print('[ ] ')
    geojson_files = [p for p in DIR_VECTOR_GEOJSON.glob('*.geojson') if not p.name.startswith('._')]
    step2_done = len(geojson_files) > 0
    attacked_dirs = [d for d in DIR_VECTOR_GEOJSON_ATTACKED.glob('*') if not d.name.startswith('.')]
    step3_done = len(attacked_dirs) > 0 and all(len(list(ad.glob('*.geojson'))) > 0 for ad in attacked_dirs if ad.is_dir())
    original_graphs = [p for p in DIR_GRAPH_ORIGINAL.glob('*_graph.pkl') if not p.name.startswith('.')]
    attacked_graphs = [p for p in DIR_GRAPH_ATTACKED.glob('*/*_graph.pkl') if not p.name.startswith('.')]
    step4_done = len(original_graphs) > 0 and len(attacked_graphs) > 0
    watermark_files = [p for p in DIR_ZEROWM.glob('*_watermark.npy') if not p.name.startswith('.')]
    step5_done = len(watermark_files) > 0
    nc_files = [p for p in DIR_RESULTS.glob('*.csv') if not p.name.startswith('.')] + [p for p in DIR_RESULTS.glob('*.xlsx') if not p.name.startswith('.')] + [p for p in DIR_RESULTS.glob('*.png') if not p.name.startswith('.')]
    step6_done = len(nc_files) > 0
    print(f'  Step2 (GeoJSON): {" " if step2_done else " "} ({len(geojson_files)} files)')
    print(f'  Step3 (Attacked): {" " if step3_done else " "} ({len(attacked_dirs)} dirs)')
    print(f'  Step4 (Graphs): {" " if step4_done else " "} (Original: {len(original_graphs)}, Attacked: {len(attacked_graphs)})')
    print(f'  Step5 (Watermarks): {" " if step5_done else " "} ({len(watermark_files)} files)')
    print(f'  Step6 (NC Results): {" " if step6_done else " "} ({len(nc_files)} files)')
    return {'step2': step2_done, 'step3': step3_done, 'step4': step4_done, 'step5': step5_done, 'step6': step6_done}


def main():
    print('=== Fig9： ===')
    existing = check_existing_files()
    if existing['step6']:
        print('\n[ ] NC .'); step6_evaluate_nc(); print('=== ==='); return
    if existing['step5'] and existing['step4']:
        print('\n[ ] .'); step6_evaluate_nc(); print('=== ==='); return
    if existing['step4']:
        print('\n[ ] .'); step5_generate_zero_watermark(); step6_evaluate_nc(); print('=== ==='); return
    if existing['step3']:
        print('\n[ ] .')
        original_geojsons = [p for p in DIR_VECTOR_GEOJSON.glob('*.geojson') if not p.name.startswith('._')]
        attacked_map: Dict[str, Dict[str, Path]] = {}
        for attacked_dir in DIR_VECTOR_GEOJSON_ATTACKED.glob('*'):
            if attacked_dir.is_dir():
                base = attacked_dir.name; attacked_map[base] = {}
                for key, _s, _lbl in FLIP_CONFIGS:
                    geojson_file = attacked_dir / f'{key}.geojson'
                    if geojson_file.exists(): attacked_map[base][key] = geojson_file
        step4_convert_to_graph(original_geojsons, attacked_map); step5_generate_zero_watermark(); step6_evaluate_nc(); print('=== ==='); return
    if existing['step2']:
        print('\n[ ] GeoJSON .')
        original_geojsons = [p for p in DIR_VECTOR_GEOJSON.glob('*.geojson') if not p.name.startswith('._')]
        attacked_map = step3_generate_flip_attacks(original_geojsons)
        step4_convert_to_graph(original_geojsons, attacked_map); step5_generate_zero_watermark(); step6_evaluate_nc(); print('=== ==='); return
    print('\n[ ] ...')
    inputs = step1_discover_inputs(); original_geojsons = step2_convert_to_geojson(inputs)
    attacked_map = step3_generate_flip_attacks(original_geojsons)
    step4_convert_to_graph(original_geojsons, attacked_map); step5_generate_zero_watermark(); step6_evaluate_nc(); print('=== ===')
    sys.exit(0)


if __name__ == '__main__':
    main()
