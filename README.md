# Capsule-network-for-fault-diagnosis

maxtrain>99%
maxtest>98%


After the original vibration signal is sampled and normalized by sliding window, it becomes a 32x32 image, and then after data enhancement, it is input into the capsule network. This is the code implementation of pure broken capsule network. Since the parameters of the capsule network are about 8558848, it takes a long time to train on my 970m GPU. But the accuracy is still very high.


General combination model can achieve high accuracy, but it needs to redesign the size of convolution kernel and use one-dimensional convolution. All of these need to be implemented by debug, which has not been implemented yet. Now capsule network fault diagnosis is generally combined with inception or bilstm.

![d490c20a8b8e342dae4ef07581fc746](https://user-images.githubusercontent.com/19371493/124874395-478a5c00-dffa-11eb-9424-1fd74a29c83c.png)
![04e93ea00110a44995f9b23ee81b4dc](https://user-images.githubusercontent.com/19371493/124874399-49ecb600-dffa-11eb-8276-7d35dfc83a48.png)
![209e0c251cd0d65a4aa918540c14c1f](https://user-images.githubusercontent.com/19371493/124874423-4f4a0080-dffa-11eb-95c3-744e39b9f5d8.png)
![0bc9c0c303ba9881ba1252682c5f172](https://user-images.githubusercontent.com/19371493/124874440-540eb480-dffa-11eb-88de-3957207370c9.png)
![1beeadd38fadf34998ffd97a47f0b03](https://user-images.githubusercontent.com/19371493/124874460-5a049580-dffa-11eb-9a76-c5abf236e853.png)

# Welcome to cite it:
```html
@ARTICLE{9761239,
  author={Chen, Biao and Liu, Tingting and He, Chao and Liu, Zecheng and Zhang, Li},
  journal={IEEE Sensors Journal}, 
  title={Fault Diagnosis for Limited Annotation Signals and Strong Noise Based on Interpretable Attention Mechanism}, 
  year={2022},
  volume={22},
  number={12},
  pages={11865-11880},
  doi={10.1109/JSEN.2022.3169341}}
 ```
# Our other work:

```html
@article{ZHANG2022110242,  
title = {Fault diagnosis for small samples based on attention mechanism},  
journal = {Measurement},  
volume = {187},  
pages = {110242},  
year = {2022},  
issn = {0263-2241},  
doi = {https://doi.org/10.1016/j.measurement.2021.110242 },  
url = {https://www.sciencedirect.com/science/article/pii/S0263224121011507},  
author = {Xin Zhang and Chao He and Yanping Lu and Biao Chen and Le Zhu and Li Zhang}  
} 
```

```html
@ARTICLE{9374403,  
author={Luo, Hao and He, Chao and Zhou, Jianing and Zhang, Li},  
journal={IEEE Access},   
title={Rolling Bearing Sub-Health Recognition via Extreme Learning Machine Based on Deep Belief Network Optimized by Improved Fireworks},   
year={2021},  
volume={9},  
number={},  
pages={42013-42026},  
doi={10.1109/ACCESS.2021.3064962},  
url = {https://ieeexplore.ieee.org/document/9374403},  
}
```
# Contact
- Chao He
- 22110435#bjtu.edu.cn   (please replace # by @)
