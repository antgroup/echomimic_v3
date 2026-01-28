<p align="center">
  <img src="asset/EchoMimicV3_logo.png.jpg"  height=60>
</p>

<h1 align='center'>EchoMimicV3: 13亿参数即可实现统一多模态、多任务人体动画生成</h1>

<div align='center'>
    <a href='https://github.com/mengrang' target='_blank'>孟让</a><sup>1</sup>&emsp;
    <a href='https://github.com/' target='_blank'>王艳</a>&emsp;
    <a href='https://github.com/' target='_blank'>吴伟鹏</a>&emsp;
    <a href='https://github.com/' target='_blank'>郑若冰</a>&emsp;
    <a href='https://lymhust.github.io/' target='_blank'>李宇明</a><sup>2</sup>&emsp;
    <a href='https://openreview.net/profile?id=~Chenguang_Ma3' target='_blank'>马晨光</a><sup>2</sup>
</div>
<div align='center'>
支付宝终端技术部，蚂蚁集团
</div>
<p align='center'>
    <sup>1</sup>项目负责人&emsp;
    <sup>2</sup>通讯作者
</p>
<div align='center'>
    <a href='https://github.com/antgroup/echomimic_v3'><img src='https://img.shields.io/github/stars/antgroup/echomimic_v3?style=social'></a>
    <a href='https://antgroup.github.io/ai/echomimic_v3/'><img src='https://img.shields.io/badge/项目主页-blue'></a>
    <a href='https://arxiv.org/abs/2507.03905'><img src='https://img.shields.io/badge/论文-Arxiv-red'></a>
    <a href='https://huggingface.co/BadToBest/EchoMimicV3'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-模型-yellow'></a>
    <a href='https://modelscope.cn/models/BadToBest/EchoMimicV3'><img src='https://img.shields.io/badge/ModelScope-模型-purple'></a>
    <a href='https://github.com/antgroup/echomimic_v3/blob/main/asset/wechat_group.png'><img src='https://badges.aleen42.com/src/wechat.svg'></a>
    <a href='https://github.com/antgroup/echomimic_v3/discussions/18'><img src='https://img.shields.io/badge/中文版-常见问题汇总-orange'></a>
    <!--<a href='https://antgroup.github.io/ai/echomimic_v2/'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Demo-yellow'></a>-->
    <!--<a href='https://antgroup.github.io/ai/echomimic_v2/'><img src='https://img.shields.io/badge/ModelScope-Demo-purple'></a>-->
    <!-- <a href='https://openaccess.thecvf.com/content/CVPR2025/papers/Meng_EchoMimicV2_Towards_Striking_Simplified_and_Semi-Body_Human_Animation_CVPR_2025_paper.pdf'><img src='https://img.shields.io/badge/Paper-CVPR2025-blue'></a> -->
    
</div>
<!-- <div align='center'>
    <a href='https://github.com/antgroup/echomimic_v3/discussions/0'><img src='https://img.shields.io/badge/English-常见问题-orange'></a>
    <a href='https://github.com/antgroup/echomimic_v3/discussions/1'><img src='https://img.shields.io/badge/中文版-常见问题汇总-orange'></a>
</div> -->

<p align="center">
  <img src="asset/algo_framework.png"  height=700>
</p>

## &#x1F4E3; 更新日志
[2026.01.22] 🔥 我们更新了 EchoMimicV3-flash-pro 模型和代码
- 🚀 8 步即可生成高质量结果，远超 V3-preview 表现
- 🧩 无需人脸位置掩码，流程更简洁，鲁棒性更强
- 💾 需 **12G** 显存
- ✅ 支持最大 768×768 分辨率生成
* [2025.08.12] 🔥 **仅需12G显存生成视频**, 量化版本[GradioUI](https://github.com/antgroup/echomimic_v3/blob/main/app_mm.py)发布。 查看 [教程](https://www.bilibili.com/video/BV1W8tdzEEVN)。感谢@[gluttony-10](https://github.com/gluttony-10)贡献。
* [2025.08.12] 🔥 EchoMimicV3支持16G显存，使用[ComfyUI](https://github.com/smthemex/ComfyUI_EchoMimic). 感谢 @[smthemex](https://github.com/smthemex)的贡献。
* [2025.08.10] 🔥 [GradioUI](https://github.com/antgroup/echomimic_v3/blob/main/app.py) 已发布，感谢 @[gluttony-10](https://github.com/gluttony-10) 的贡献。
* [2025.08.09] 🔥 我们在 ModelScope 上发布了 [模型](https://modelscope.cn/models/BadToBest/EchoMimicV3)。
* [2025.08.08] 🔥 我们在 Huggingface 上发布了 [代码](https://github.com/antgroup/echomimic_v3) 和 [模型](https://huggingface.co/BadToBest/EchoMimicV3)。
* [2025.07.08] 🔥 我们的 [论文](https://arxiv.org/abs/2507.03905) 在 arxiv 上公开。

## &#x1F305; 示例展示
<p align="center">
  <img src="asset/echomimicv3.jpg"  height=1000>
</p>
<table class="center">
<tr>
    <td width=100% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/f33edb30-66b1-484b-8be0-a5df20a44f3b" muted="false"></video>
    </td>
</tr>
<tr>
    <td width=100% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/056105d8-47cd-4a78-8ec2-328ceaf95a5a" muted="false"></video>
    </td>
</tr>
</table>


### 中文驱动效果
<table class="center">
<tr>
    <td width=25% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/fc1ebae4-b571-43eb-a13a-7d6d05b74082" muted="false"></video>
    </td>
    <td width=25% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/54607cc7-944c-4529-9bef-715862ba330d" muted="false"></video>
    </td>
    <td width=25% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/4d1de999-cce2-47ab-89ed-f2fa11c838fe" muted="false"></video>
    </td>
    <td width=25% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/41e701cc-ac3e-4dd8-b94c-859261f17344" muted="false"></video>
    </td>
</tr>
</table>

更多演示视频，请访问[项目主页](https://antgroup.github.io/ai/echomimic_v3/)

## 快速开始
### 环境配置
- 测试系统环境：Centos 7.2/Ubuntu 22.04, Cuda >= 12.1
- 测试 GPU：A100(80G) / RTX4090D (24G) / V100(16G)
- 测试 Python 版本：3.10 / 3.11

### 🛠️ 安装
#### 1. 创建 conda 环境
```
conda create -n echomimic_v3 python=3.10
conda activate echomimic_v3
```

#### 2. 安装其他依赖
```
pip install -r requirements.txt
```
### 🧱 模型准备

| 模型名称        |                       下载链接                                           |    备注                      |
| --------------|-------------------------------------------------------------------------------|-------------------------------|
| Wan2.1-Fun-1.3B-InP  |      🤗 [Huggingface](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP)       | 基础模型
| wav2vec2-base |      🤗 [Huggingface](https://huggingface.co/facebook/wav2vec2-base-960h)          | 音频编码器
| EchoMimicV3-preview      |      🤗 [Huggingface](https://huggingface.co/BadToBest/EchoMimicV3)              | preview的权重
| EchoMimicV3-preview      |      🤗 [ModelScope](https://modelscope.cn/models/BadToBest/EchoMimicV3)              | preview的权重
| EchoMimicV3-flash-pro      |      🤗 [Huggingface](https://huggingface.co/BadToBest/EchoMimicV3/tree/main/echomimicv3-flash-pro)              | flash-pro的权重

-- EchoMimicV3-flash-pro**权重** 文件组织如下：
```
./flash-pro/
├── Wan2.1-Fun-V1.1-1.3B-InP
├── chinese-wav2vec2-base
└── transformer
    └── diffusion_pytorch_model.safetensors
```
-- EchoMimicV3-preview**权重** 文件组织如下：
```
./preview/
├── Wan2.1-Fun-V1.1-1.3B-InP
├── wav2vec2-base-960h
└── transformer
    └── diffusion_pytorch_model.safetensors
```
### 🔑 快速推理 EchoMimicV3-flash-pro
```
bash run_flash_pro.sh
```
### 🔑 快速推理 EchoMimicV3-preview
```
python infer_preview.py
```
### 🔑 快速推理preview量化版本
```
python app_mm.py
```
#### 提示
- 音频 CFG：音频 CFG `audio_guidance_scale` 最佳范围为 2~3。增加音频 CFG 值可以改善唇同步效果，减少音频 CFG 值可以提高视觉质量。- 文本 CFG：文本 CFG `gu
- idance_scale` 最佳范围为 3~6。增加文本 CFG 值可以更好地遵循提示词，减少文本 CFG 值可以提高视觉质量。
- TeaCache：`teacache_threshold` 的最佳范围为 0~0.1。
- 采样步数：头部动画为 5 步，全身动作为 15~25 步。
- ​长视频生成：如果需要生成超过 138 帧的视频，可以使用长视频 CFG。
- 尝试降低`partial_video_length`节省显存。


## 📝 待办事项
| 状态 | 里程碑                                                                |     
|:--------:|:-------------------------------------------------------------------------|
|    ✅    | EchoMimicV3 推理代码已发布至 GitHub   | 
|    ✅   | EchoMimicV3-preview 模型已发布至 HuggingFace | 
|    ✅   | EchoMimicV3-preview 模型已发布至 ModelScope | 
|    🚀  | 在线体验版 | 
|    🚀    | 英文和中文预训练模型（Preview 版本）已发布至 ModelScope   | 
|    🚀    | 英文和中文预训练模型（720P）已发布至 HuggingFace | 
|    🚀    | 英文和中文预训练模型（720P）已发布至 ModelScope   | 
|    🚀    | EchoMimicV3 训练代码已发布至 GitHub   | 

## &#x1F680; EchoMimic 系列
* EchoMimicV3: 13亿参数即可实现统一多模态、多任务人体动画生成。[GitHub](https://github.com/antgroup/echomimic_v3)
* EchoMimicV2: 面向震撼、简化、半身人体动画生成。[GitHub](https://github.com/antgroup/echomimic_v2)
* EchoMimicV1: 基于可编辑关键点条件的逼真语音驱动肖像动画生成。[GitHub](https://github.com/antgroup/echomimic)


## &#x1F4D2; 引用

如果我们的工作对您的研究有帮助，请引用我们的论文：

```
@misc{meng2025echomimicv3,
  title={EchoMimicV3: 1.3B Parameters are All You Need for Unified Multi-Modal and Multi-Task Human Animation},
  author={Rang Meng, Yan Wang, Weipeng Wu, Ruobing Zheng, Yuming Li, Chenguang Ma},
  year={2025},
  eprint={2507.03905},
  archivePrefix={arXiv}
}
```
## 📜 许可证
本仓库中的模型采用 Apache 2.0 许可证。我们不对您生成的内容主张任何权利，
赋予您自由使用的权利，但您的使用需遵守该许可证的规定。
您需对模型的使用负全责，不得用于违反法律法规、伤害个人或群体、传播有害个人信息、散布虚假信息或针对弱势群体的行为。

## &#x1F31F; Star 历史
[![Star History Chart](https://api.star-history.com/svg?repos=antgroup/echomimic_v3&type=Date)](https://www.star-history.com/#antgroup/echomimic_v3&Date)
