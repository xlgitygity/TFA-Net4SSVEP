# TFA-Net

This repository contains the implementation of our paper titled **"Decoding SSVEP via Calibration-Free TFA-Net: A Novel Network Using Time-Frequency Features."**
![TFA-Net Architecture](TFA-Net4SSVEP/figure/Fig.%202.png)

## Dataset

The dataset used for this implementation can be downloaded from the following link:

- [Benchmark Dataset for SSVEP-based Brain-Computer Interfaces](https://bci.med.tsinghua.edu.cn/download.html)

### Citation

If you use this dataset in your research, please cite the following paper:

"Wang Y, Chen X, Gao X, Gao S. A benchmark dataset for SSVEP-based brain–computer interfaces. IEEE Transactions on Neural Systems and Rehabilitation Engineering. 2016 Nov 10;25(10):1746-52."


## Usage

- **Preprocessing**: Use `CWT_preprocess` to transform SSVEP signals into time-frequency representations. Since the CWT process is time-consuming, it is recommended to store the processed data instead of performing real-time CWT. The transformed data can occupy a large amount of memory, so you can adjust the `overlap` parameter in the `slice_and_cwt` function to control the data augmentation factor and alleviate excessive memory usage.
