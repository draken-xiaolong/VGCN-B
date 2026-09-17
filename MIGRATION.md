# 换电脑恢复与验证

Git 仓库包含代码、选定模型权重、标准化器、Fig. 5 的原始绘图数据。
原始地图另存为 `VGCN-B-original-data.zip`，必须同时带走；只克隆代码不够。
`VGCN-B-private-paper-backup.zip` 保存最终原稿、修改稿、拒稿信、参考 PDF、
图件、原始训练权重与修订工具，不属于公开代码仓库。

## 1. 恢复代码和地图

安装 Git、Python 3.13（64 位）；合约测试另外需要 Node.js 和 npm。
在新电脑任意目录执行：

```powershell
git clone https://github.com/draken-xiaolong/VGCN-B.git
cd VGCN-B
# 替换为实际保存位置；压缩包里的目录直接合并进仓库根目录
Expand-Archive -LiteralPath 'D:\Backup\VGCN-B-original-data.zip' -DestinationPath .
python verify_inputs.py
```

应检查到 98 个文件（46 张训练地图、43 张测试地图、6 张复合实验地图、
权重、标准化器和水印），且没有缺失或变化。
也可离线从备份克隆：`git clone D:\Backup\VGCN-B-code.bundle VGCN-B`。
不要在旧电脑上直接复制 Python 虚拟环境；在新电脑重新安装。

## 2. 安装 CPU 环境

以下路径适用于 Windows PowerShell，不需要激活脚本或 NVIDIA 显卡：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu-win-py313.lock.txt
.\.venv\Scripts\python.exe -m pip check
```

两条安装命令必须先后完成。GPU 可另建环境安装适配驱动的 PyTorch；
硬件与底层数值库不同可能影响临界二值化结果，不能预先保证逐字节一致。
锁定文件保存本次 Windows / Python 3.13 CPU 验证的完整依赖版本。
其他操作系统可从 `requirements-cpu.txt` 安装，但未在本次迁移检查中实测。

## 3. 验证可运行部分

一条命令执行下面全部 Python 步骤：
`.\.venv\Scripts\python.exe -X utf8 run_reproduction.py`。
它在失败时停止，并将每一步日志和状态保存至
`ReproductionAudit/portable_run/`；合约测试仍需单独运行。
使用 `--quick` 可仅运行无需地图的训练冒烟检查和 Fig. 5 重绘。

```powershell
.\.venv\Scripts\python.exe -X utf8 smoke_train.py
.\.venv\Scripts\python.exe -X utf8 reproduction_audit.py
.\.venv\Scripts\python.exe -X utf8 reproduce_compound.py
.\.venv\Scripts\python.exe -X utf8 key_scrambling_audit.py
.\.venv\Scripts\python.exe -X utf8 figures/fig5/generate_fig5.py
cd BlockChain
npm ci
npm test
```

参考记录：43 张测试地图全部完成；同描述符对 11 个；原名义阈值 0.90 下，
1806 次定向非匹配验证中接受 369 次。固定注册时间的六地图复合实验均值约
0.902804（清空旧图缓存后，独立 CPU 环境与原 GPU 环境一致）。先前的
0.914331 使用了遗留图缓存，不能作为从头复现的结果。Fig. 5 重绘使用
历史 CSV，不等于重新完成五种对比方法的全部实验。
训练 smoke test 使用合成数据；从零完整训练、所有基线和消融尚未完成复现。
原始基线作者的独立示例脚本可能仍含其本机路径，不在上述已验证入口范围内。

## 4. 恢复论文

将私人论文备份解压到独立工作目录。最终原稿位于 `Paper/Final`，
最新工作修改稿位于 `Paper/Revision_20260917`。安装完整 MiKTeX 或 TeX Live，
进入修改稿目录后依次运行：

```text
pdflatex -interaction=nonstopmode -halt-on-error A1_Manuscript.tex
bibtex A1_Manuscript
pdflatex -interaction=nonstopmode -halt-on-error A1_Manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error A1_Manuscript.tex
```

修订稿仍存在投稿资格和实验验证待办，详见 `REVISION_TRACKER.md`。
云存储密钥、钱包私钥、`.env` 未收进任何迁移包；线上服务需自行重新配置。
备份压缩包仍在旧电脑上时，不算完成异机迁移：必须复制到移动硬盘、
本人网盘或获授权的远程存储，并用 `SHA256.json` 核对哈希后再停用旧电脑。
