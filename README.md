## Sketch Your Own GAN论文复现
## 如何训练：

参考options.py中的args，可以按照要求设定超参数，运行train.py即可开始训练。

如果需要基于已有的模型生成图像，则直接运行generate.py即可生成图像，可以参考其中的args设定参数。

如果需要做评估，直接进入run_metrics.py即可，参数也在其中可以得到。



## 如何获取数据：

对于作者给出的基础数据，可通过运行data内的脚本得到，并用lsun解压。如果需要斯坦福数据集可直接访问[ai.stanford.edu/~jkrause/cars/car_dataset.html](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)获取。

如果需要二次元数据集，则可在以下百度网盘链接得到：

链接：https://pan.baidu.com/s/1jP-SGdTUE6OGleU9pLM0dQ 
提取码：7460 

我们的预训练模型，如猫马等除了通过pretrained内的脚本下载，也可通过访问[GitHub - NVlabs/stylegan2: StyleGAN2 - Official TensorFlow Implementation](https://github.com/NVlabs/stylegan2)直接得到预训练模型并使用作者给出的convert转换器将其转换为对应pt或pth文件。

我们的二次元预训练模型则可通过以下链接得到，也需要convert转换。

链接：https://pan.baidu.com/s/1E1ZOkgaqQlKIU-owgH5BbQ 
提取码：njd6 

而我们的eval自制数据集如下下载得到：

链接：https://pan.baidu.com/s/1LknkI2flolpgbVwczTjYQw 
提取码：oikq 



