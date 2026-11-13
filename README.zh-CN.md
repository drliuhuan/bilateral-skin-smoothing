# bilateral-skin-smoothing

[![CI](https://github.com/drliuhuan/bilateral-skin-smoothing/actions/workflows/ci.yml/badge.svg)](https://github.com/drliuhuan/bilateral-skin-smoothing/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)

[English](README.md) · 中文

一个自包含的皮肤磨皮参考实现（纯 NumPy，无深度学习训练依赖）：多尺度**双边网格**
边缘保持平滑 + 细节层混合 + 锐化/柔化 + 色调调整，可选 **U-2-Net（Apache-2.0）**
自动人物蒙版。

主要特点：

* **双边网格平滑**：splat → 可分离 `(1,2,1)/4` 模糊 → ΣW 归一化 → 三线性 slice，
  复杂度 O(N)，大图也快，属于快速双边滤波近似；
* **多尺度**：由粗到细的金字塔，逐层两遍链，细节层（fine / medium / large）加权混合，
  带软阈值保护五官边缘；
* **锐化/柔化**：软限幅的 unsharp（`x^(2/3)` 拐点）+ 1/2 回混；
* **色调**：warmth / brightness / contrast；
* **自动蒙版**：U-2-Net 人物 matte × 经典肤色项（HSV + HSL 样条），
  没有 OpenCV/模型时自动回退 HSV。

依赖只有 NumPy 和 Pillow；OpenCV 仅用于可选的人物蒙版。

> 本项目是独立实现，与任何商业产品无隶属关系，也未使用其代码或模型权重。

## 安装

### 从 PyPI

```bash
pip install bilateral-skin-smoothing
pip install "bilateral-skin-smoothing[mask]"   # 可选：U-2-Net 蒙版依赖
```

> 包已准备好发布；PyPI 发布由 `.github/workflows/publish.yml`（trusted publishing）
> 在 GitHub Release 时自动完成，需先在 PyPI 项目设置中配置。未发布前请用下面的方式安装。

### 从 GitHub

```bash
pip install git+https://github.com/drliuhuan/bilateral-skin-smoothing.git
```

### 从 Release wheel 或本地构建

```bash
python -m pip install build
python -m build            # 生成 dist/bilateral_skin_smoothing-*.whl 与 .tar.gz
pip install dist/*.whl
```

### 源码运行

```bash
git clone https://github.com/drliuhuan/bilateral-skin-smoothing.git
cd bilateral-skin-smoothing
pip install -r requirements.txt
```

## 快速开始

```python
import skin_smoothing as P

img, meta = P.load_image("portrait.jpg")        # float64 [0,1]，8/16-bit RGB
res = P.smooth_skin(
    img,
    smoothing=75,          # 0..100 总强度
    threshold=50,          # 0..100 细节/边缘阈值
    fine=50, medium=50, large=50,
    sharpness=0.0, softness=0.0,
    warmth=0.0, brightness=0.0, contrast=0.0,
    portrait_size="medium",
)
P.save_image("portrait_smoothed.png", res["output"], bit_depth=meta["bit_depth"])

res["mask"]        # 实际使用的蒙版
res["base"]        # 平滑后的基底
res["details"]     # fine / medium / large 细节层
```

### 自动蒙版（U-2-Net）

```bash
pip install "bilateral-skin-smoothing[mask]"   # 或 pip install opencv-python-headless
bss-fetch-model                                # 下载到 ~/.cache/bilateral-skin-smoothing/
```

源码 checkout 下运行 `python scripts/fetch_u2net.py` 会保存到 `models/`。
查找顺序：`u2net_path=` 参数 → `U2NET_MODEL_PATH` 环境变量 → 模块旁/当前目录的
`models/` → 用户缓存。

之后 `smooth_skin` 默认使用 **U-2-Net-lite 人物 matte × 经典肤色项**；
没有 OpenCV 或模型文件时自动回退 HSV，流程仍可运行。

```python
res = P.smooth_skin(img, mask_method="hsv")   # 强制经典回退
mask = P.build_mask(img)                      # 单独构建蒙版
pm = P.u2net_person_matte(img)                # 不可用时返回 None
```

自定义模型路径：`export U2NET_MODEL_PATH=/path/to/u2netp.onnx`（或传 `u2net_path=`）。

## 处理流程

```
读取图像（8/16-bit RGB）
  → 自动蒙版：人物 matte × 肤色项，羽化
  → 多尺度双边网格：
       splat（空间 × 亮度分箱，12-bit）
       → 可分离 (1,2,1)/4 模糊（两轴）
       → 按 ΣW 归一化
       → 三线性 slice 回全分辨率
  → 由粗到细层间合成（12-bit lerp）
  → 细节层混合 + 边缘软阈值
  → 按蒙版混回（不透明度）
  → 锐化 / 柔化（软限幅 unsharp + 1/2 回混）
  → 色调调整
  → 保存图像（8/16-bit RGB）
```

数学细节见 [`docs/algorithm.md`](docs/algorithm.md)。

## 演示与测试

```bash
python demo.py                    # 合成人像 → out/*.png 与 out/metrics.json
python tests/test_primitives.py   # 27/27 通过
```

演示输出包含前后对比、蒙版、细节层、滑块扫描、16-bit 对比与滤波变体；
没有 OpenCV/模型时测试自动覆盖 HSV 回退路径。

## 项目结构

```
skin_smoothing.py          核心实现（可导入模块）
synthetic.py               确定性合成测试图
demo.py                    演示脚本
docs/algorithm.md          算法数学说明
models/README.md           可选 U-2-Net 权重说明（不随仓库分发权重）
scripts/fetch_u2net.py     模型下载脚本（含 SHA-256 校验）
tests/test_primitives.py   27 项单元测试
.github/workflows/ci.yml        CI：测试（py3.9–3.13）+ 构建 + wheel 冒烟测试
.github/workflows/publish.yml   GitHub Release 时经 trusted publishing 发布 PyPI
pyproject.toml              打包元数据（setuptools）
```

## 版本与发布

1. 修改 `pyproject.toml` 的 `version` 和 `skin_smoothing.py` 的 `__version__`；
2. 提交、打 tag 并创建 GitHub Release（`vX.Y.Z`）；
3. 发布工作流会构建 sdist/wheel 并通过 trusted publishing 上传 PyPI
   （需先在 PyPI 项目设置中配置环境）。

本地构建检查：

```bash
python -m pip install build twine
python -m build
twine check dist/*
python -m pip install dist/*.whl
```

## 许可证

Apache-2.0，见 [`LICENSE`](LICENSE)。

可选的 U-2-Net 权重同样是 **Apache-2.0**，不随仓库分发；再分发时请保留其许可证与声明。
