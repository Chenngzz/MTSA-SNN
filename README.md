# MTSA-SNN: A Multi-modal Time Series Analysis Model Based on Spiking Neural Network
Time series analysis and modelling constitute a
crucial research area. However, traditional artificial neural net-
works often encounter challenges when dealing with complex,
non-stationary time series data, such as high computational
complexity, limited ability to capture temporal information,
and difficulty in handling event-driven data. To address these
challenges, we propose a Multi-modal Time Series Analysis
Model Based on Spiking Neural Network (MTSA-SNN). The
Pulse Encoder unifies the encoding of temporal images and
sequential information in a common pulse-based representation.
The Joint Learning Module employs a joint learning function
and weight allocation mechanism to fuse information from multi-
modal pulse signals complementary. Additionally, we incorporate
wavelet transform operations to enhance the model’s ability to
analyze and evaluate temporal information. Experimental results
demonstrate that our method achieved superior performance on
three complex time-series tasks. This work provides an effective
event-driven approach to overcome the challenges associated with
analyzing intricate temporal information
## Requirements

- [PyTorch](https://pytorch.org/) >= 1.10.1
- [Python](https://www.python.org/) >= 3.7
- [Einops](https://github.com/arogozhnikov/einops) = 0.6.1
- [NumPy](https://numpy.org/) = 1.24.3
- [TorchVision](https://pytorch.org/vision/stable/transforms.html) = 0.9.1+cu111
- [scikit-learn](https://scikit-learn.org/stable/index.html) = 1.2.2
- [CUDA](https://developer.nvidia.com/cuda-toolkit) >= 11.3

# MTSA_SNN Overall Model
![MTSA_SNN Overall Model](https://github.com/Chenngzz/MTSA-SNN/blob/main/image/SNN_net.png)

# Model Structure
<img src="https://github.com/Chenngzz/MTSA-SNN/blob/main/image/workflow.png" width="600" height="350">

# Wavelet Transform
MTSP-SNN employs wavelet transform to decompose input
signals into four subbands: LL, LH, HH and HL, which
represent distinct signal characteristics in terms of different
frequencies and spatial scale. Specifically, the LL subband contains the low-frequency components of the signal. In contrast, the LH and HH subbands capture the high-frequency components of both low and high-frequency signals, respectively, corresponding to different signal frequencies. The HL subband contains the low-frequency components of high-frequency signals.
<img src="https://github.com/Chenngzz/MTSA-SNN/blob/main/image/Wavelet%20Transform.png" width="600" height="500">
![猫图片](https://github.com/Chenngzz/MTSA-SNN/blob/main/image/heatmap.png) ![狗图片](https://example.com/dog.jpg)

* 斜体 *
** 粗体 **
[链接](http://www.example.com)
![图像](http://www.example.com/image.jpg)
- 列表项目
