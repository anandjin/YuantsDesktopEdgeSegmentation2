# YuantsDesktopEdgeSegmentation

## 1. 项目简介

此项目是AIDesktopagent项目(一种人工智能桌面代理，能够通过屏幕流与计算机系统交互，并通过光标和键盘动作提供输出

AI为你提供母语级高精翻译)的分支项目,用于处理计算机视觉相关问题:

从个人电脑执行电脑视觉分析截图, 包括但不限于:

1. 边缘检测, 文字、以线为基础的图标识别, 自然图像识别, 用于分割的直线识别, 图像分割
2. 利用api和相关论文中的现有模型完成传统的CV任务

## 2. 配置安装说明

1. 克隆该存储库或下载脚本到本地机器:

   ```python
   git clone https://github.com/your_username/your_repository.git
   cd your_repository
   ```

2. 创建并激活虚拟环境（推荐）：

   ```python
   python -m venv venv
   source venv/bin/activate  # 对于Windows用户，使用 `venv\Scripts\activate`
   ```

3. 依赖库

   确保已安装以下库： 

   numpy 

   pillow 

   scikit-learn

   你可以使用以下命令安装这些依赖项：

```python
pip install -r requirements.txt
```

## 3. 使用方法

要运行该脚本，请按照以下步骤操作：

1. 打开终端或命令提示符，并导航到脚本所在的目录
2. 使用 Python 运行脚本：

```python
python edge.py
```

运行该脚本后，将执行以下步骤：

1. 在延迟3秒后捕获屏幕截图并保存为 `screenshot0.png`

2. 使用简单差分法检测截图中的边缘

3. 保存边缘检测结果

4. 分类检测到的边缘元素，包括文本/图标、直线和不规则大形状(自然图像)

5. 从截图中提取文本和图标

6. 为直线添加标签

7. 为不规则大形状添加红色边框

8. 获取边缘图信息

9. 合并所有信息并生成带标签的可视化图像

10. 保存带标签的可视化图像为 `labeled_screenshot0.png`

## 4. 贡献指南



## 5. 联系信息

