# Application of band combination in landslide  identification of remote sensing images driven by deep learning

![Github stars](https://img.shields.io/github/stars/ubuntu-achieve/BCLSI.svg)![Github stars](https://img.shields.io/github/directory-file-count/ubuntu-achieve/BCLSI.svg)![Github stars](https://img.shields.io/github/license/ubuntu-achieve/BCLSI.svg)

----

[README_en](./README.md)

## 环境配置

通过运行代码 `pip install -r requirements.txt` 完成Python环境的配置

## 模型训练

通过修改文件 `configs/swin_tiny_patch4_window4_128_lite.yaml` 中的 `IN_CHANS` 项分别进行原始数据和波段组合数据的模型训练