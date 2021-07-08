# capsule-network-for-fault-diagnosis

maxtrain>99%
maxtest>98%


将原始振动信号经过滑动窗口采样与归一化之后，变为32x32的图像，然后经过数据增强，在输入胶囊网络中去，这是纯碎的胶囊网络的代码实现。由于胶囊网络的参数量大约为8558848，在我的970m的gpu上训练时间很久。但是准确率还是很高的。
After the original vibration signal is sampled and normalized by sliding window, it becomes a 32x32 image, and then after data enhancement, it is input into the capsule network. This is the code implementation of pure broken capsule network. Since the parameters of the capsule network are about 8558848, it takes a long time to train on my 970m GPU. But the accuracy is still very high.


一般组合模型可以取得很高的准确率，但是需要重新设计卷积核的大小，使用一维卷积。这都需要debug来实现，暂时还没有实现。现在胶囊网络的故障诊断一般是和inception或者bilstm结合了。
General combination model can achieve high accuracy, but it needs to redesign the size of convolution kernel and use one-dimensional convolution. All of these need to be implemented by debug, which has not been implemented yet. Now capsule network fault diagnosis is generally combined with inception or bilstm.

目前还可以玩得点就是和attention机制结合一下了。

![d490c20a8b8e342dae4ef07581fc746](https://user-images.githubusercontent.com/19371493/124874395-478a5c00-dffa-11eb-9424-1fd74a29c83c.png)
![04e93ea00110a44995f9b23ee81b4dc](https://user-images.githubusercontent.com/19371493/124874399-49ecb600-dffa-11eb-8276-7d35dfc83a48.png)
![209e0c251cd0d65a4aa918540c14c1f](https://user-images.githubusercontent.com/19371493/124874423-4f4a0080-dffa-11eb-95c3-744e39b9f5d8.png)
![0bc9c0c303ba9881ba1252682c5f172](https://user-images.githubusercontent.com/19371493/124874440-540eb480-dffa-11eb-88de-3957207370c9.png)
![1beeadd38fadf34998ffd97a47f0b03](https://user-images.githubusercontent.com/19371493/124874460-5a049580-dffa-11eb-9a76-c5abf236e853.png)
