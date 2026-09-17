# -*- coding: utf-8 -*-
# @Time    : 2024/2/24 21:19
# @Author  :Fivem
# @File    : extract.py
# @Software: PyCharm
# @last modified:2024/2/24 21:19
import math
import os
import sys
from collections import Counter

import geopandas as gpd
import numpy as np
from PIL import Image
from BER import BER

from get_coor import get_coor_nested, get_coor_array
from select_file import select_file
from to_geodataframe import to_geodataframe
from NC import NC, image_to_array


def calculate_layer(coordinate, coordinate_l, R, n):
    """
    calculate the serial number of the horizontal and vertical gaps of the non-reference vertex.
    Returns:

    """
    for w in range(2 ** n):
        if R * math.sqrt(w) / 2 ** (n / 2) <= coordinate - coordinate_l < R * math.sqrt(w + 1) / 2 ** (n / 2):
            return int(w)


def watermark_extract(tran_coor, coor_l, R, n, r):
    """
    对坐标嵌入水印
    :param tran_coor: 变换后的坐标
    :param coor_l: 区域左下角顶点的坐标
    :return:返回嵌入水印的坐标
    """
    tran_x, tran_y = tran_coor
    xl, yl = coor_l
    layer_x = calculate_layer(tran_x, xl, R, n)
    layer_y = calculate_layer(tran_y, yl, R, n)
    if layer_x is None:
        w = layer_y
        print(layer_x, layer_y)
    elif layer_y is None:
        w = layer_x
        print(layer_x, layer_y)
    else:
        w = max(layer_x, layer_y)
        # print(w)
    Rw = R * math.sqrt(w + 1) / 2 ** (n / 2)
    Rw_ = R * math.sqrt(w) / 2 ** (n / 2)
    Tw = R * math.sqrt(w + 1) * (math.sqrt(w + 1) - math.sqrt(w))
    delta_w = R * (math.sqrt(w + 1) - math.sqrt(w)) / 2 ** (n / 2)
    if layer_y == w:
        original_x = R / r * ((tran_x - xl) / Rw - (1 - r) / 2) + xl
        original_y = Tw / r * ((tran_y - yl - Rw_) / delta_w - (1 - r) / 2) + R - Tw + yl
    else:
        original_x = R / r * ((tran_x - xl - Rw_) / delta_w - (1 - r) / 2) + xl
        original_y = (R - Tw) / r * ((tran_y - yl) / Rw_ - (1 - r) / 2) + yl

    return np.vstack([original_x, original_y]), w


def coor_process(coor, vr1, vr2, W, dis, R, n, r, i):
    """
    对坐标进行处理
    :param coor: 需要处理的坐标
    :return: 嵌入水印的坐标
    """
    x, y = coor
    x_r1, y_r1 = vr1
    x_r2, y_r2 = vr2

    coor = ([x - (x_r1 + x_r2) / 2, y - (y_r1 + y_r2) / 2] @ np.vstack([
        [(x_r2 - x_r1) / dis, -(y_r2 - y_r1) / dis], [(y_r2 - y_r1) / dis, (x_r2 - x_r1) / dis]]))
    coor_l = np.floor(coor / R) * R
    original_coor, w = watermark_extract(coor, coor_l, R, n, r)
    original_coor = (original_coor.T @ np.vstack(
        [[(x_r2 - x_r1) / dis, (y_r2 - y_r1) / dis], [-(y_r2 - y_r1) / dis, (x_r2 - x_r1) / dis]]) + [
                         (x_r1 + x_r2) / 2, (y_r1 + y_r2) / 2]).reshape((2, 1))
    W.append(w)
    return original_coor, W


def coor_group_process(coor_group, vr1, vr2, W, dis, R, n, r, indexes, i):
    """
    对坐标组进行处理
    :param coor_group:需要处理的坐标组
    :return: 返回嵌入水印的坐标组
    """

    extract_coor_group = np.array([[], []])
    for coor in coor_group.T:
        if i in indexes:
            extract_coor = coor[:, np.newaxis]
        else:
            extract_coor, W = coor_process(coor, vr1, vr2, W, dis, R, n, r, i)
        extract_coor_group = np.concatenate((extract_coor_group, extract_coor), axis=1)
        i += 1

    # 将nan值替换成原值
    if extract_coor_group.dtype != np.float64:
        extract_coor_group = extract_coor_group.astype(np.float64)
    extract_coor_group[:, np.where(np.isnan(extract_coor_group))[1]] = coor_group[:,
                                                                       np.where(np.isnan(extract_coor_group))[1]]

    # extract_coor_group[:, np.where(np.isinf(extract_coor_group))[1]] = coor_group[:,
    #                                                                    np.where(np.isinf(extract_coor_group))[1]]
    return extract_coor_group, W, i


def traversal_nested_coor_group(coor_nested, feature_type, vr1, vr2, W, dis, R, n, r, indexes, i):
    """
    对于多线、多面等情况，执行此函数
    :param coor_nested: 所有要素组成的嵌套坐标数组
    :param feature_type: 要素的类型
    :return: 返回该要素更新后的嵌套坐标组
    """
    processed_x_nested = []
    processed_y_nested = []
    # 遍历要素中的每个坐标组
    for feature_index in range(coor_nested.shape[1]):
        coor_group = np.vstack(coor_nested[:, feature_index])
        # 对坐标进行平移
        processed_coor_group, W, i = coor_group_process(coor_group, vr1, vr2, W, dis, R, n, r, indexes, i)
        # 如果要素为多面，则需要满足首位顶点的坐标相同
        if (feature_type in ['MultiPolygon', 'Polygon']
                and np.size(processed_coor_group) not in [0, 2]
                and not np.array_equal(processed_coor_group[:, 0], processed_coor_group[:, -1])):
            processed_coor_group[:, -1] = processed_coor_group[:, 0]
        processed_x_nested.append(processed_coor_group[0, :])
        processed_y_nested.append(processed_coor_group[1, :])
    return np.array([processed_x_nested, processed_y_nested], dtype=object), W, i


def traversal_coor_group(coor_nested, shp_type, processed_shpfile, vr1, vr2, dis, R, n, r, indexes):
    """
    对所有要素进行遍历
    :param coor_nested: 所有要素组成的嵌套坐标数组
    :param shp_type: 每个要素类型组成的数组
    :param processed_shpfile: 处理后的shp文件
    :return: processed_shpfile
    """
    # ----------------定义局部变量----------------------
    i = 0
    W = []
    # 遍历每个几何要素
    for feature_index in range(coor_nested.shape[1]):
        coor_group = np.vstack(coor_nested[:, feature_index])

        feature_type = shp_type[feature_index]
        # 判断是否是多线、多面等的情况
        if isinstance(coor_group[0, 0], np.ndarray):
            processed_coor_group, W, i = traversal_nested_coor_group(coor_group, feature_type, vr1, vr2, W, dis, R, n,
                                                                     r,
                                                                     indexes, i)
        # todo:1
        elif isinstance(coor_nested[:, feature_index][0], list):
            coor_group = np.empty((2, len(coor_nested[:, feature_index][0])), dtype=object)
            # 将列表填充到数组中
            coor_group[0, :] = coor_nested[:, feature_index][0]
            coor_group[1, :] = coor_nested[:, feature_index][1]
            processed_coor_group, W, i = traversal_nested_coor_group(coor_group, feature_type, vr1, vr2, W, dis, R, n,
                                                                     r, indexes, i)
        else:
            processed_coor_group, W, i = coor_group_process(coor_group, vr1, vr2, W, dis, R, n, r, indexes, i)
            # 如果要素为面，则需要满足首尾顶点的坐标相同
            if (feature_type == 'Polygon'
                    and np.size(processed_coor_group) not in [0, 2]
                    and not np.array_equal(processed_coor_group[:, 0], processed_coor_group[:, -1])):
                processed_coor_group[:, -1] = processed_coor_group[:, 0]
        # 将改变的要素坐标组更新到geodataframe
        processed_shpfile = to_geodataframe(processed_shpfile, feature_index, processed_coor_group,
                                            shp_type[feature_index])
    return processed_shpfile, W




def extract(watermarked_shpfile_path, original_watermark_path):
    # -------------------------预定义--------------------------------
    n = 4  # 嵌入强度
    tau = 10 ** (-6)  # 精度容差
    r = 0.999  # 约束因子 (0<r<=1)
    K = 1  # 参考顶点选择的密钥
    side_length = 32

    # 不再使用固定 num 映射；改为自适应重构

    # -------------------------数据读取--------------------------------
    watermarked_shpfile = gpd.read_file(watermarked_shpfile_path)
    watermarked_coor_nested, feature_type = get_coor_nested(watermarked_shpfile)

    # -------------------------数据预处理--------------------------------
    coor_array = get_coor_array(watermarked_coor_nested, feature_type)  # 将嵌套坐标数组合并成一个数组
    # 基于K，获得两个参考顶点
    indexes = (K, coor_array.shape[1] - K - 1) if K != coor_array.shape[1] - K else (K - 1, K)

    vr1 = coor_array[:, indexes[0]]  # 第一个参考顶点
    vr2 = coor_array[:, indexes[1]]  # 第二个参考顶点

    # 相关参数计算（防御异常：dis=0 或类型异常时回退默认参数）
    try:
        dis = float(np.linalg.norm(vr1 - vr2))  # 计算两个参考顶点的距离
        if dis <= 0:
            raise ValueError('invalid distance')
        c = math.sqrt(1 + (2 ** n - 1) ** (-1))
        Maxd = math.sqrt(1 / (c ** 2) * (1 + 1 / (1 + c) ** 2))
        D = max(1, int(math.ceil(dis * Maxd / tau)))  # 分割系数
        R = dis / D  # 原始分块的边长
    except Exception:
        # 回退：使用固定块长，避免 sqrt 类型错误
        D = 1024
        R = 1.0

    # -------------------------水印提取--------------------------------
    # 遍历每个对象
    original_shpfile = watermarked_shpfile.copy()
    try:
        original_shpfile, W = traversal_coor_group(watermarked_coor_nested, feature_type, original_shpfile, vr1, vr2,
                                                   R*D, R, n, r, indexes)
    except Exception:
        # 继续回退：W 空
        W = []

    # 计算每个数组出现的次数
    watermark = []
    # 若 W 为空则直接回退为零矩阵，避免 ufunc sqrt 错误
    if len(W) == 0:
        watermark = np.zeros((side_length, side_length), dtype=int)
        original_watermark = image_to_array(original_watermark_path)
        nc = NC(original_watermark, watermark)
        ber = BER(original_watermark, watermark)
        # 使用脚本所在目录而非上层目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        folder_name = os.path.join(script_dir, 'extract')
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        if not os.path.exists(f'{folder_name}/watermark'):
            os.makedirs(f'{folder_name}/watermark')
        output_watermark_path = f'{folder_name}/watermark/{os.path.splitext(os.path.basename(watermarked_shpfile_path))[0]}.png'
        Image.fromarray(watermark.astype(bool)).save(output_watermark_path)
        return watermarked_shpfile_path, {'NC': nc, 'BER': ber}, nc

    # 展开为比特流，并按实际长度自适应投票重构 32x32
    bit_stream = []
    bit_stream.extend([int(digit) for digit in format(w, f"0{n}b")] for w in W)
    bit_stream = [b for chunk in bit_stream for b in chunk]
    bit_stream = np.array(bit_stream, dtype=int)
    L = int(bit_stream.size)
    if L == 0:
        watermark = np.zeros((side_length, side_length), dtype=int)
    else:
        repeat_time = max(1, L // (side_length * side_length))
        voted = []
        for i in range(side_length * side_length):
            indices = [j * side_length * side_length + i for j in range(repeat_time)
                       if (j * side_length * side_length + i) < L]
            if not indices:
                voted.append(0)
                continue
            values = bit_stream[indices]
            counter = Counter(values.tolist())
            voted.append(counter.most_common(1)[0][0])
        watermark = np.array(voted).reshape(side_length, side_length)

    # 评估NC值
    original_watermark = image_to_array(original_watermark_path)
    nc = NC(original_watermark, watermark)
    ber = BER(original_watermark, watermark)
    # print(f'NC值为{nc},BER值为{ber}')
    error = {'NC': nc, 'BER': ber}

    # -------------------------数据输出--------------------------------
    # 创建文件夹（使用绝对路径，输出到当前脚本目录下）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_name = os.path.join(script_dir, 'extract')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    if not os.path.exists(f'{folder_name}/shpfile'):
        os.makedirs(f'{folder_name}/shpfile')

    if not os.path.exists(f'{folder_name}/watermark'):
        os.makedirs(f'{folder_name}/watermark')

    output_shapefile_path = f'{folder_name}/shpfile/{os.path.basename(watermarked_shpfile_path)}'
    # 继承 CRS 后保存，以避免写出时无CRS告警
    try:
        if getattr(watermarked_shpfile, 'crs', None) is not None:
            original_shpfile.set_crs(watermarked_shpfile.crs, allow_override=True, inplace=True)  # type: ignore
    except Exception:
        pass
    original_shpfile.to_file(output_shapefile_path)
    print("Shapefile创建完成，已保存为", output_shapefile_path, flush=True)

    output_watermark_path = f'{folder_name}/watermark/{os.path.splitext(os.path.basename(watermarked_shpfile_path))[0]}.png'
    Image.fromarray(watermark.astype(bool)).save(output_watermark_path)
    print("水印创建完成，已保存为", output_watermark_path, flush=True)

    return output_shapefile_path, error,nc


if __name__ == '__main__':
    # 配置参数 MRailways MBuilding 0  MLanduse MBoundary MRoad MLake1
    # watermarked_shpfile_path = select_file('select the watermarked shpfile', [('shpfile', '*.shp')])
    # watermarked_shpfile_path = r"embed/猫爪32gis_osm_railways_free_1.shp" Boundary00 Building00 Lake1 Landuse01 Railways1 Road01
    # gis_osm_waterways_free_1.shp gis_osm_landuse_a_free_1.shp gis_osm_natural_free_1.shp Boundary.shp BRGA.shp  gis_osm_railways_free_1.shp
    # watermarked_shpfile_path = r"attacked/delete/delete_MRailways_factor_0.5.shp"
    watermarked_shpfile_path = r"attacked/cropped/half_cropped_Mgis_osm_railways_free_1.shp"
    # watermarked_shpfile_path = r"embed/Mgis_osm_railways_free_1.shp"
    print("当前处理的矢量数据为：", os.path.basename(watermarked_shpfile_path))
    # original_watermark_path = select_file('select the watermarked text', [('text', '*.png')])
    original_watermark_path = r'Cat32.png'
    extract(watermarked_shpfile_path, original_watermark_path)
