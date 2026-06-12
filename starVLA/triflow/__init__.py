"""TriFlow: 单塔全生成式三模态 (V/L/A) 扩散框架.

零 LLM / 零 VLM / 零预训练语言权重:
- V: 冻结 DINOv3-S 特征 (复用 jointflow 离线预计算管线)
- L: BPE 分词 (只用分词文件) + 从头学习的 embedding 表
- A: StateActionTransform 归一化动作 chunk
- state: 完全不输入

一条序列 [V | L | A | V'] 进同一个 from-scratch transformer:
干净块 = 条件, 加噪块 = 目标(各自独立 timestep), 任意子集 → 任意子集
只是噪声模式不同 (policy / v2l / fdm / passive / idm / joint)。

本包零修改任何现有文件; 对 jointflow v1 与 starVLA 核心一律 import 复用。
"""
