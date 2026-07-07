# Habitat-VLN 环境配置指南

**项目简介**: 本项目旨在搭建基于 Habitat-Sim 的视觉语言导航（Vision-and-Language Navigation, VLN）深度学习开发环境。本文档详细记录了从零开始在 Windows 11 + WSL2 环境下配置完整开发环境的全过程，包括环境搭建、依赖安装、版本冲突解决等关键步骤。

---

## 0. 准备工作

### 0.1 硬件配置

| 组件 | 型号 |
| :--- | :--- |
| **设备** | 联想拯救者 Y9000P 2024 |
| **CPU** | Intel Core i9-14900HX |
| **GPU** | NVIDIA GeForce RTX 4060 Laptop GPU（8GB VRAM） |

### 0.2 软件配置

| 组件 | 版本 / 说明 |
| :--- | :--- |
| **宿主机操作系统** | Windows 11 |
| **运行环境** | WSL2（Windows Subsystem for Linux） |
| **Linux 发行版** | Ubuntu 24.04 LTS |
| **项目存储位置** | `F:\habitat_vln`（物理存储于 Windows F 盘，以节省系统盘空间） |

---

## 1. 配置环境

### 1.1 配置 WSL 并安装 Ubuntu 操作系统

以 **管理员身份** 打开 **Windows PowerShell**，执行以下命令下载适用于 Linux 的 Windows 子系统：

```bash
wsl --install
```

**注意**: 执行后控制台会显示"请求的操作成功。直到重新启动系统前更改将不会生效。"，此时请 **立即重启电脑**。

重启电脑后，再次以 **管理员身份** 打开 **Windows PowerShell**，执行相同的命令以安装 Ubuntu 操作系统：

```bash
wsl --install
```

![[fd5ebef6ebbe0a0c8d6455eed88d9bce.png]]

安装完毕后，系统会自动弹出"欢迎使用 WSL"的窗口。

![[30f1696f479673a6c548f70f394350ad.png]]

回到 PowerShell 控制台窗口，根据提示新建一个账号和密码。**请注意**：在输入密码过程中，屏幕不会显示任何输入的字符，属于正常现象，请完全盲打输入。

![[69e37cf914022dc28ad434c338411141.png]]

按下回车键，进入 Linux 环境。

![[27fb70437bbbd9faad5107aa93c86d45.png]]

至此，Linux 环境配置成功。

---

### 1.2 创建项目文件夹

首先，切换到 Windows 的 F 盘（或其他您希望存放项目的位置）：

```bash
cd ..
cd /mnt/f
```

![[d3b415a8a67d425a574362c2f517c758.png]]

在 F 盘根目录下创建项目文件夹 `habitat_vln`：

```bash
mkdir habitat_vln
```

进入刚刚创建的项目文件夹：

```bash
cd habitat_vln
```

此时，您可以通过以下命令在 Windows 资源管理器中直接打开该项目文件夹，方便后续文件管理：

```bash
explorer.exe .
```

![[9e79af2355eb93124150fd579ebae398.png]]

---

### 1.3 安装 Miniconda

下载 Miniconda 安装包：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
```

静默安装，将其安装到 Linux 用户目录下的专属目录：

```bash
bash miniconda.sh -b -u -p ~/miniconda3
```

![[f9c8c91c9b8b1fd3b3072cf88cdce723 1.png]]

清理掉刚才下载的安装包：

```bash
rm miniconda.sh
```

将 Conda 写入系统启动项，使其在每次打开终端时自动激活：

```bash
~/miniconda3/bin/conda init bash
```

刷新配置文件，使改动立即生效：

```bash
source ~/.bashrc
```

![[380ed5d6fff08fbe659494972accee80.png]]

---

### 1.4 部署并激活 Python 虚拟环境

创建一个名为 `habitat` 的虚拟环境，并指定 Python 3.9 版本：

```bash
conda create -n habitat python=3.9 -y
```

**注意**: 如果遇到 `Terms of Service have not been accepted` 的协议拦截提示，请先执行以下命令手动签署服务条款：

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

签署完成后，重新执行创建环境的命令：

```bash
conda create -n habitat python=3.9 -y
```

![[994c943acc08029a04f997057717bca7.png]]

![[e5efeec84f1125f63745e462d4575160.png]]

激活虚拟环境 `habitat`：

```bash
conda activate habitat
```

**提示**: 激活成功后，终端命令行最左边的 `(base)` 会变为 `(habitat)`，表明虚拟环境已成功激活。

![[e93f915e59bf784411881fd38db3330b.png]]

执行以下命令，查看显卡信息，确认 GPU 驱动正常：

```bash
nvidia-smi
```

![[ac49e1eb5c91252f76c62d42818139d5.png]]

---

### 1.5 安装所需库 PyTorch、Habitat-Sim

#### 1.5.1 安装 PyTorch（含 CUDA 12.1 GPU 加速）

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

![[27f824250429a16ab3eaf83dd2baacc3.png]]

![[ec7a30ad3338c246b7be9e0c2598c2bd.png]]

![[2dc863069271d3b0028f48184fde5a5b.png]]

#### 1.5.2 安装 Habitat-Sim（含 Bullet 物理引擎支持）

```bash
conda install -c aihabitat -c conda-forge habitat-sim withbullet -y
```

![[796ec1dafd262ae2fbd4312e28642c83.png]]

![[0ef2be406b419ceafcc8d513c4a459f4.png]]

#### 1.5.3 安装系统级依赖库

为防止后续 Python 代码运行时报错，需要安装 WSL2 环境下缺失的系统库。

首先，安装 `libgomp1`（OpenMP 多线程支持库）：

```bash
sudo apt update && sudo apt install libgomp1 -y
```

![[6b7c80a20a36641dcb47c597c586223a.png]]

![[0d863eddef3f7b26b1bcafac031cb176.png]]

#### 1.5.4 验证 PyTorch 与 CUDA

进入 Python 环境进行验证：

```bash
python
```

测试 `torch` 以及 CUDA 是否可用：

```python
import torch
print(torch.cuda.is_available())
```

**预期输出**: 若屏幕打印 `True`，则说明 PyTorch 部署成功，且 GPU 加速功能正常。

![[991718c05de6816139c8b1a3403b03e2.png]]

#### 1.5.5 安装 OpenGL 运行库

为防止调用 `habitat_sim` 时报错，安装 OpenGL 运行库：

```bash
sudo apt install libopengl0 -y
```

![[0ff623821be4961ec91de9135f3b4c00.png]]

#### 1.5.6 解决 Numba 版本兼容性冲突

由于 `numba` 库版本过新，与 `habitat-sim` 存在兼容性冲突，需要卸载高版本并手动安装兼容的稳定版本：

```bash
pip uninstall numba -y
pip install numba==0.58.1
```

![[29788d5c88bfb0f9fdc8301976d59b52 1.png]]

#### 1.5.7 补齐其他依赖包

安装其他提示报错的依赖包：

```bash
pip install imageio-ffmpeg pillow==10.4.0
```

![[f9cda433b9b14020fad53bd310b8dbfa.png]]

#### 1.5.8 验证 Habitat-Sim

安装完成后，再次进入 Python 环境进行验证：

```bash
python
```

测试 `habitat_sim` 是否安装成功：

```python
import habitat_sim
print(habitat_sim.__version__)
```

**预期输出**: 打印版本号 `0.3.3`，表示 Habitat-Sim 安装成功。

![[d0ece708442327233edca4674f481873.png]]

---

## 2. 技术栈概览

| 组件 | 版本 | 作用 |
| :--- | :--- | :--- |
| **Python** | 3.9.25 | 核心开发语言 |
| **Conda** | Miniconda（最新版） | 环境隔离与包管理 |
| **PyTorch** | 2.x（CUDA 12.1） | 深度学习框架 |
| **Habitat-Sim** | 0.3.3 | 3D 物理仿真环境 |
| **Numba** | 0.58.1 | 科学计算加速（核心兼容库） |

---

## 3. 常见问题与解决方案

### 3.1 Anaconda 新协议拦截（TOS Error）

- **现象**: 创建环境时报 `Terms of Service have not been accepted`。
- **原因**: 最新版 Anaconda 要求用户手动签署服务条款。
- **解决**: 使用 `conda tos accept --override-channels --channel <URL>` 手动签署服务条款。

### 3.2 导入 Torch 报错 `libgomp.so.1` 缺失

- **现象**: `import torch` 触发 `OSError`。
- **原因**: WSL2 环境缺少 OpenMP 多线程支持库。
- **解决**: 执行 `sudo apt install libgomp1 -y`。

### 3.3 导入 Habitat-Sim 报错 `libOpenGL.so.0` 缺失

- **现象**: `import habitat_sim` 提示无法打开 OpenGL 共享库。
- **原因**: WSL2 环境缺少 OpenGL 运行库。
- **解决**: 执行 `sudo apt install libopengl0 -y`。

### 3.4 Numba 属性错误（AttributeError）

- **现象**: `habitat-sim` 启动时因 `numba.core.types` 缺失属性而崩溃。
- **原因**: `numba` 最新版本与 `habitat-sim` 存在 API 兼容性问题。
- **解决**: 将 `numba` 降级至 `0.58.1`，并同步安装兼容的 `numpy 1.26.4`。

---

## 4. 快速验证

完成上述所有步骤后，您可以使用以下命令快速验证环境是否配置成功：

```bash
# 激活环境
conda activate habitat

# 验证 CUDA 与物理引擎
python -c "import torch; print(torch.cuda.is_available()); import habitat_sim; print(habitat_sim.__version__)"
```

**预期输出**: `True` 且版本号为 `0.3.3`。
