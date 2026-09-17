#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
水印图像置乱工具
基于数据集名称和时间戳生成置乱密钥，用于置乱和恢复水印图像
"""

import numpy as np
import hashlib
import time


def generate_scramble_key(dataset_name, timestamp=None):
    """
    根据数据集名称和时间戳生成置乱密钥
    
    Args:
        dataset_name: 数据集名称
        timestamp: 时间戳（如果为None，使用当前时间）
    
    Returns:
        scramble_key: 置乱密钥（整数）
    """
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
    
    # 使用密钥确定迭代次数（取模确保在合理范围内）
    # Arnold变换的周期性：对于N×N图像，周期最大为N²
    iterations = (scramble_key % (N * N)) + 1
    
    scrambled = image.copy()
    
    # Arnold变换矩阵参数
    a, b = 1, 1
    
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
    
    # Arnold逆变换矩阵参数
    # 逆变换：[x', y'] = [2*x - y, -x + y] mod N
    
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
    """
    保存置乱信息到文件
    
    Args:
        dataset_name: 数据集名称
        scramble_key: 置乱密钥
        timestamp: 时间戳
        output_path: 输出文件路径
    """
    import json
    
    info = {
        'dataset_name': dataset_name,
        'scramble_key': int(scramble_key),
        'timestamp': int(timestamp)
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def load_scramble_info(info_path):
    """
    从文件加载置乱信息
    
    Args:
        info_path: 信息文件路径
    
    Returns:
        dataset_name, scramble_key, timestamp
    """
    import json
    
    with open(info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)
    
    return info['dataset_name'], info['scramble_key'], info['timestamp']


# 测试代码
if __name__ == '__main__':
    # 创建测试图像
    test_image = np.random.randint(0, 2, (32, 32), dtype=np.uint8)
    
    # 生成置乱密钥
    dataset_name = "test_dataset"
    scramble_key, timestamp = generate_scramble_key(dataset_name)
    print(f"数据集: {dataset_name}")
    print(f"时间戳: {timestamp}")
    print(f"置乱密钥: {scramble_key}")
    
    # 置乱图像
    scrambled = scramble_image(test_image, scramble_key)
    print(f"\n原始图像与置乱图像是否相同: {np.array_equal(test_image, scrambled)}")
    
    # 恢复图像
    recovered = descramble_image(scrambled, scramble_key)
    print(f"原始图像与恢复图像是否相同: {np.array_equal(test_image, recovered)}")
    
    # 验证恢复的正确性
    if np.array_equal(test_image, recovered):
        print("\n✓ 置乱和恢复测试通过！")
    else:
        print("\n✗ 置乱和恢复测试失败！")
