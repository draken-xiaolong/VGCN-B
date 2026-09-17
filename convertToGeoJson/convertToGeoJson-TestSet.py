#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
矢量地图数据转换为GeoJSON格式（仅测试集）
功能：将 SourceData/TestSet 目录下的 GDB 和 SHP 文件转换为 GeoJSON 格式
输出：转换后的文件保存在 GeoJson/TestSet 目录下
"""

from fiona.drvsupport import supported_drivers
from convertToGeoJson import VectorToGeoJsonConverter, logger


def main():
    """主函数：仅转换 TestSet 数据集"""
    try:
        # 确保支持 GDB 读取
        supported_drivers["OpenFileGDB"] = "r"

        # 创建转换器
        converter = VectorToGeoJsonConverter()

        logger.info("开始矢量数据转换（仅 TestSet 数据集）...")
        logger.info("本次运行将完全覆盖之前生成的 TestSet GeoJSON 文件")

        # 只转换 TestSet
        test_converted, test_failed = converter.convert_dataset("TestSet", converter.test_output_dir)

        # 输出统计信息
        logger.info("=== TestSet 转换完成 ===")
        logger.info(f"TestSet 成功转换: {test_converted} 个文件")
        logger.info(f"TestSet 转换失败: {test_failed} 个文件")
        logger.info(f"TestSet 输出目录: {converter.test_output_dir}")

        # 列出生成的 TestSet 文件
        converter._list_generated_files("TestSet", converter.test_output_dir)

    except KeyboardInterrupt:
        logger.info("用户中断操作")
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        raise


if __name__ == "__main__":
    main()
