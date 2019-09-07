# [ResNet](https://github.com/tfzoo/ResNet) 
## [ResNet简介](https://github.com/tfzoo/ResNet/wiki) 

ResNet（Residual Neural Network）是广泛应用的CNN特征提取网络，参数量比VGGNet低，推广性非常好，可以直接用到InceptionNet网络中。

卷积网络或者全连接网络在信息传递的时候或多或少会存在信息丢失，损耗等问题，同时还有导致梯度消失或者梯度爆炸，导致很深的网络无法训练。所以深度CNN网络达到一定深度后再一味地增加层数并不能带来进一步地分类性能提高，反而会招致网络收敛变得更慢，test dataset的分类准确率也变得更差。排除数据集过小带来的模型过拟合等问题后，我们发现过深的网络仍然还会使分类准确度下降，例如VGG网络达19层后再增加层数分类性能的下降。

ResNet的主要思想是在网络中增加了直连通道，即Highway Network的思想。此前的网络结构是性能输入做一个非线性变换，而Highway Network则允许保留之前网络层的一定比例的输出。ResNet的思想和Highway Network的思想也非常类似，允许原始输入信息直接传到后面的层中。ResNet最大的区别在于有很多的旁路将输入直接连接到后面的层，这种结构也被称为shortcut或者skip connections。

ResNet有不同的网络层数，比较常用的是50-layer，101-layer，152-layer。

###  [天府动物园 tfzoo：tensorflow models zoo](http://www.tfzoo.com)
####   qitas@qitas.cn
[![sites](tfzoo/tfzoo.png)](http://www.tfzoo.com)