#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量测试所有矢量地图的零水印系统
对pso_data下的矢量文件进行完整测试
"""

import os
import cv2
import numpy as np
import time
from Zero_watermarking import *
from Extract_zero_watermarking import XOR2, Arnold_Decrypt
from NC import NC  # 使用用户提供的NC计算方法

# 获取脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def batch_test_zero_watermark():
    """批量测试所有矢量地图的零水印"""
    
    # 矢量文件列表（pso_data下的新矢量地图）
    vector_files = [
        'gis_osm_landuse_a_free_1.shp',
        'gis_osm_pois_free_1.shp',
        'gis_osm_railways_free_1.shp',
        'gis_osm_waterways_free_1.shp',
        'BRGA.shp',
        'LRDL.shp',
        'RESA.shp',
        'RESP.shp'
    ]
    
    # 水印图片
    watermark_img = os.path.join(BASE_DIR, 'Cat32.png')
    
    # 创建必要的输出文件夹
    zero_watermark_dir = os.path.join(BASE_DIR, 'zero_watermark')
    extract_dir = os.path.join(BASE_DIR, 'extract')
    os.makedirs(zero_watermark_dir, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)
    
    print("🚀 开始批量测试矢量地图零水印系统")
    print("=" * 60)
    
    # 创建结果存储
    results = []
    
    for i, vector_file in enumerate(vector_files, 1):
        print(f"\n🔄 [{i}/{len(vector_files)}] 测试 {vector_file}")
        print("-" * 40)
        
        vector_path = os.path.join(BASE_DIR, 'pso_data', vector_file)
        if not os.path.exists(vector_path):
            print(f"❌ 文件不存在: {vector_path}")
            continue
            
        try:
            start_time = time.time()
            
            # === 步骤1: 生成零水印 ===
            print("📝 步骤1: 生成零水印...")
            img = cv2.imread(watermark_img, 0)
            img_deal = Watermark_deal(img)
            Arnold_img = Arnold_Encrypt(img_deal)
            Lst_WaterMark = Arnold_img.flatten()
            
            # 读取矢量数据
            XLst, YLst, feature_num = Read_Shapfile(vector_path)
            print(f"   特征数量: {feature_num}")
            
            # 构造特征矩阵并生成零水印
            List_Fea = Construction(XLst, feature_num, Lst_WaterMark)
            List_Zero = XOR(List_Fea, Lst_WaterMark)
            Array_Z = np.array(List_Zero).reshape(32, 32)
            
            # 保存零水印图像
            zero_watermark_file = os.path.join(zero_watermark_dir, f'{vector_file[:-4]}_zero.png')
            cv2.imwrite(zero_watermark_file, Array_Z.astype(np.uint8))  # 确保二值性质
            print(f"   ✅ 零水印已保存: {zero_watermark_file}")
            
            # === 步骤2: 提取零水印 ===
            print("📝 步骤2: 提取零水印...")
            img_zero_loaded = cv2.imread(zero_watermark_file, 0)
            img_deal_zero = Watermark_deal(img_zero_loaded)
            List_Zero_loaded = img_deal_zero.flatten()
            
            # 重新构造特征矩阵并提取水印
            List_Fea2 = Construction(XLst, feature_num, List_Zero_loaded)
            Lst_WaterMark_extract = XOR2(List_Fea2, List_Zero_loaded)
            Re_mark = np.array(Lst_WaterMark_extract).reshape(32, 32)
            Decode_image = Arnold_Decrypt(Re_mark)
            
            # 保存提取的水印
            extract_file = os.path.join(extract_dir, f'{vector_file[:-4]}_extract.png')
            cv2.imwrite(extract_file, Decode_image.astype(np.uint8))  # 确保二值性质
            print(f"   ✅ 提取水印已保存: {extract_file}")
            
            # === 步骤3: 计算NC值 ===
            # 直接比较原始Cat32.png和提取的最终水印
            nc_value = NC(img, Decode_image)
            
            # === 步骤4: 记录结果 ===
            elapsed_time = time.time() - start_time
            
            result = {
                'file': vector_file,
                'features': feature_num,
                'nc_value': nc_value,
                'time': elapsed_time,
                'success': True
            }
            results.append(result)
            
            print(f"   📊 NC值: {nc_value:.6f}")
            print(f"   ⏱️  耗时: {elapsed_time:.2f}秒")
            
            if nc_value >= 0.99:
                print(f"   🎉 测试成功！")
            else:
                print(f"   ⚠️  NC值异常")
                
        except Exception as e:
            print(f"   ❌ 测试失败: {str(e)}")
            result = {
                'file': vector_file,
                'features': 0,
                'nc_value': 0.0,
                'time': 0.0,
                'success': False,
                'error': str(e)
            }
            results.append(result)
    
    # === 输出总结报告 ===
    print("\n" + "=" * 60)
    print("📊 批量测试结果总结")
    print("=" * 60)
    
    success_count = sum(1 for r in results if r['success'] and r['nc_value'] >= 0.99)
    total_count = len(results)
    
    print(f"📈 测试概况:")
    print(f"   总测试文件数: {total_count}")
    print(f"   成功文件数: {success_count}")
    if total_count > 0:
        print(f"   成功率: {success_count/total_count*100:.1f}%")
    else:
        print(f"   成功率: N/A (无测试文件)")
    
    print(f"\n📋 详细结果:")
    print(f"{'文件名':<15} {'特征数':<8} {'NC值':<10} {'耗时(秒)':<8} {'状态'}")
    print("-" * 55)
    
    for result in results:
        if result['success']:
            status = "✅ 成功" if result['nc_value'] >= 0.99 else "⚠️ 异常"
            print(f"{result['file'][:-4]:<15} {result['features']:<8} {result['nc_value']:<10.6f} {result['time']:<8.2f} {status}")
        else:
            print(f"{result['file'][:-4]:<15} {'N/A':<8} {'N/A':<10} {'N/A':<8} ❌ 失败")
    
    # === 性能统计 ===
    if success_count > 0:
        successful_results = [r for r in results if r['success']]
        avg_nc = np.mean([r['nc_value'] for r in successful_results])
        avg_time = np.mean([r['time'] for r in successful_results])
        total_features = sum([r['features'] for r in successful_results])
        
        print(f"\n📈 性能指标:")
        print(f"   平均NC值: {avg_nc:.6f}")
        print(f"   平均处理时间: {avg_time:.2f}秒")
        print(f"   总处理特征数: {total_features:,}")
        
    print(f"\n📁 文件输出:")
    zero_count = len(os.listdir(zero_watermark_dir)) if os.path.exists(zero_watermark_dir) else 0
    extract_count = len(os.listdir(extract_dir)) if os.path.exists(extract_dir) else 0
    print(f"   零水印图像: zero_watermark/ ({zero_count}个文件)")
    print(f"   提取水印图像: extract/ ({extract_count}个文件)")
    
    if success_count == total_count:
        print(f"\n🎉 所有测试完美通过！零水印系统表现优秀！")
    else:
        print(f"\n⚠️  部分测试未通过，请检查详细信息。")

def clean_temp_files():
    """清理临时和测试文件"""
    # 切换到脚本所在目录
    os.chdir(BASE_DIR)
    
    temp_files = [
        'Zero_image.png',
        'Zero_image_plot.png', 
        'Decode_image.png',
        'watermark_extraction_results.png',
        'encrypted_boundary.shp',
        'decrypted_boundary.shp',
        'test_Zero_image.png',
        'test_Decode_image.png',
        'verify_complete_flow.py',
        'Zero_watermarking_improved.py',
        'test_real_flow.py',
        'M_22x22.png',
        '流程运行总结.md'
    ]
    
    print("\n🗑️  清理临时文件...")
    cleaned_count = 0
    for file in temp_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"   ✅ 已删除: {file}")
                cleaned_count += 1
            except Exception as e:
                print(f"   ❌ 删除失败 {file}: {e}")
    
    print(f"   📊 共清理了 {cleaned_count} 个临时文件")

if __name__ == '__main__':
    # 批量测试
    batch_test_zero_watermark()
    
    # 清理临时文件
    clean_temp_files()
    
    print("\n✨ 批量测试完成！")
