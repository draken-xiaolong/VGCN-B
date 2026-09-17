#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量运行Wu25对比实验的所有Fig测试脚本（Fig1-Fig12）
按顺序执行所有鲁棒性测试，并生成汇总报告
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
import json


class FigRunner:
    """Fig测试脚本运行器"""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.results = []
        self.start_time = None
        self.log_file = None
        
    def setup_logging(self):
        """设置日志文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = self.script_dir / 'logs'
        log_dir.mkdir(exist_ok=True)
        self.log_file = log_dir / f'run_all_{timestamp}.log'
        
    def log(self, message, level='INFO'):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"
        print(log_message)
        
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_message + '\n')
    
    def run_fig_script(self, fig_name):
        """运行单个Fig脚本"""
        script_path = self.script_dir / f'{fig_name}.py'
        
        if not script_path.exists():
            self.log(f"脚本不存在: {script_path}", 'WARNING')
            return {
                'fig': fig_name,
                'status': 'SKIP',
                'reason': '脚本文件不存在',
                'duration': 0
            }
        
        self.log(f"{'='*80}")
        self.log(f"开始运行: {fig_name}")
        self.log(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            # 运行脚本
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.script_dir),
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=3600  # 1小时超时
            )
            
            duration = time.time() - start_time
            
            # 记录输出
            if result.stdout:
                self.log(f"\n{fig_name} 输出:\n{result.stdout}")
            
            if result.stderr:
                self.log(f"\n{fig_name} 错误:\n{result.stderr}", 'WARNING')
            
            # 判断执行状态
            if result.returncode == 0:
                status = 'SUCCESS'
                self.log(f"{fig_name} 执行成功 (耗时: {duration:.2f}秒)", 'SUCCESS')
            else:
                status = 'FAILED'
                self.log(f"{fig_name} 执行失败 (返回码: {result.returncode})", 'ERROR')
            
            return {
                'fig': fig_name,
                'status': status,
                'returncode': result.returncode,
                'duration': duration,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            self.log(f"{fig_name} 执行超时 (超过1小时)", 'ERROR')
            return {
                'fig': fig_name,
                'status': 'TIMEOUT',
                'duration': duration,
                'reason': '执行超时'
            }
            
        except Exception as e:
            duration = time.time() - start_time
            self.log(f"{fig_name} 执行异常: {e}", 'ERROR')
            return {
                'fig': fig_name,
                'status': 'ERROR',
                'duration': duration,
                'error': str(e)
            }
    
    def generate_summary_report(self):
        """生成汇总报告"""
        self.log(f"\n{'='*80}")
        self.log("执行汇总报告")
        self.log(f"{'='*80}\n")
        
        total_duration = time.time() - self.start_time
        success_count = sum(1 for r in self.results if r['status'] == 'SUCCESS')
        failed_count = sum(1 for r in self.results if r['status'] == 'FAILED')
        error_count = sum(1 for r in self.results if r['status'] == 'ERROR')
        timeout_count = sum(1 for r in self.results if r['status'] == 'TIMEOUT')
        skip_count = sum(1 for r in self.results if r['status'] == 'SKIP')
        
        self.log(f"总执行时间: {total_duration:.2f}秒 ({total_duration/60:.2f}分钟)")
        self.log(f"总测试数: {len(self.results)}")
        self.log(f"成功: {success_count}")
        self.log(f"失败: {failed_count}")
        self.log(f"错误: {error_count}")
        self.log(f"超时: {timeout_count}")
        self.log(f"跳过: {skip_count}")
        
        self.log(f"\n详细结果:")
        self.log(f"{'-'*80}")
        
        for result in self.results:
            status_symbol = {
                'SUCCESS': '✓',
                'FAILED': '✗',
                'ERROR': '⚠',
                'TIMEOUT': '⏱',
                'SKIP': '⊘'
            }.get(result['status'], '?')
            
            duration_str = f"{result['duration']:.2f}s" if result['duration'] > 0 else "N/A"
            self.log(f"{status_symbol} {result['fig']:10s} - {result['status']:8s} ({duration_str})")
        
        # 保存JSON报告
        report_path = self.script_dir / 'logs' / f'run_all_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration': total_duration,
                'summary': {
                    'total': len(self.results),
                    'success': success_count,
                    'failed': failed_count,
                    'error': error_count,
                    'timeout': timeout_count,
                    'skip': skip_count
                },
                'results': self.results
            }, f, indent=2, ensure_ascii=False)
        
        self.log(f"\nJSON报告已保存: {report_path}")
        self.log(f"日志文件: {self.log_file}")
        
        # 返回是否全部成功
        return success_count == len(self.results)
    
    def run_all(self):
        """运行所有Fig脚本"""
        self.setup_logging()
        self.start_time = time.time()
        
        self.log("="*80)
        self.log("开始批量运行Wu25对比实验 Fig1-Fig12测试脚本")
        self.log("="*80)
        self.log(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"工作目录: {self.script_dir}")
        self.log("")
        
        # Fig脚本列表
        fig_scripts = [
            'Fig1', 'Fig2', 'Fig3', 'Fig4', 'Fig5', 'Fig6',
            'Fig7', 'Fig8', 'Fig9', 'Fig10', 'Fig11', 'Fig12'
        ]
        
        # 依次运行每个脚本
        for i, fig_name in enumerate(fig_scripts, 1):
            self.log(f"\n进度: [{i}/{len(fig_scripts)}]")
            result = self.run_fig_script(fig_name)
            self.results.append(result)
            
            # 如果失败，继续执行下一个
            if result['status'] in ['FAILED', 'ERROR']:
                self.log(f"\n{fig_name} 执行失败，继续执行下一个...", 'WARNING')
        
        # 生成汇总报告
        all_success = self.generate_summary_report()
        
        self.log(f"\n{'='*80}")
        if all_success:
            self.log("所有测试执行成功！", 'SUCCESS')
        else:
            self.log("部分测试执行失败，请查看日志了解详情。", 'WARNING')
        self.log(f"{'='*80}")
        
        return all_success


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='批量运行Wu25对比实验 Fig1-Fig12测试脚本')
    parser.add_argument('--yes', '-y', action='store_true', help='跳过确认，直接运行')
    args = parser.parse_args()
    
    print("""
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║          Wu25对比实验 Fig1-Fig12 批量测试运行器                ║
║                                                                ║
║  此脚本将依次运行所有鲁棒性测试脚本                            ║
║  预计总耗时: 30-60分钟（取决于数据集大小）                     ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    # 确认是否继续
    if not args.yes:
        try:
            response = input("是否开始运行所有测试？(y/n): ").strip().lower()
            if response not in ['y', 'yes', '是']:
                print("已取消执行。")
                return
        except KeyboardInterrupt:
            print("\n已取消执行。")
            return
    
    print("\n开始执行...\n")
    
    # 创建运行器并执行
    runner = FigRunner()
    success = runner.run_all()
    
    # 返回退出码
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
