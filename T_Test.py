import os
import cv2
import numpy as np
from scipy import stats

"""
T检验要求：
1. 已知总体均值
2. 可得到一个样本均值及该样本标准差
3. 样本服从正态分布或近似服从正态分布
此处采用配对T检验。由于滑坡区域大小并不严格服从正态分布，但我们共采用759组数据，根据中心极限定理，当数据量大于30时，无论样本源自何种分布，其样本均值总近似服从正态分布。
"""
# H0假设：是否水分指数和植被指数对模型预测结果没有影响
# H1假设：是否水分指数和植被指数对模型预测结果有影响
# unet_14, unet_16 = "pre_unet_14", "pre_unet_16"
unet_14, unet_16 = "pre_swin_unet_14", "pre_swin_unet_16"
unet_list_14, unet_list_16 = [], []
for file in os.listdir(unet_14):
    img = cv2.imread(os.path.join(unet_14, file))
    unet_list_14.append(np.sum(img))
    img = cv2.imread(os.path.join(unet_16, file))
    unet_list_16.append(np.sum(img))
pre_training, post_training = np.array(unet_list_14), np.array(unet_list_16)

tstat, pval = stats.ttest_rel(post_training, pre_training)

# Display results
print("t-stat: {:.2f}   pval: {:.4f}".format(tstat, pval))
# 若P值小于0.05则可拒绝原假设，认为加入水分指数和植被指数后预测结果存在显著差异
