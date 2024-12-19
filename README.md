# TFA-Net

This repository contains the implementation of our paper：

Lei Xu, Xinyi Jiang, Ruimin Wang, Pan Lin, Yuankui Yang, Yue Leng, Wenming Zheng, and Sheng Ge, ["Decoding SSVEP via Calibration-Free TFA-Net: A Novel Network Using Time-Frequency Features."](https://ieeexplore.ieee.org/document/10777482), published in IEEE Journal of Biomedical and Health Informatics (JBHI).

![TFA-Net Architecture](TFA-Net4SSVEP/figure/Fig.%202.png)

## Dataset

The dataset used for this implementation can be downloaded from the following link:

- [Benchmark Dataset for SSVEP-based Brain-Computer Interfaces](https://bci.med.tsinghua.edu.cn/download.html)

### Citation

If you use this dataset in your research, please cite the following paper:

"Wang Y, Chen X, Gao X, Gao S. A benchmark dataset for SSVEP-based brain–computer interfaces. IEEE Transactions on Neural Systems and Rehabilitation Engineering. 2016 Nov 10;25(10):1746-52."


## Usage

- **Preprocessing**: Use `CWT_preprocess.py` to transform SSVEP signals into time-frequency representations. Since the CWT process is time-consuming, it is recommended to store the processed data instead of performing real-time CWT. The transformed data can occupy a large amount of memory, so you can adjust the `overlap` parameter in the `slice_and_cwt` function to control the data augmentation factor and alleviate excessive memory usage.

- **Training and Testing the Model**：Run the `kFoldCrossVal.py` file to perform cross-validation and train and test the model. You can choose between the TFA-Net in `Proposed_Model.py` or the comparison models in `Control_Model.py`.

## Wavelet Transform Details

The Continuous Wavelet Transform (CWT) is used to convert SSVEP signals into time-frequency representations in this project. And you can find more details and resources about wavelet transforms on the [PyWavelets website](https://pywavelets.readthedocs.io/en/latest/index.html#)
