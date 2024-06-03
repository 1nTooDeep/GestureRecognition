# 使用说明

​	本项目基于深度学习实现了手势识别。使用者可以在屏幕前做出特定手势进行识别。训练数据使用Jester数据集

下载地址：[https://developer.qualcomm.com/software/ai-datasets/jester]

## 项目包简介

`config`项目的配置文件，加载一个`config.json`文件，相应的配置可以在`config.json`中查看。

`dataset`加载Jester数据集的数据，通过读取`config`对象的内容进行加载。

`model`是项目的模型包，可以在`model`中加载相应的模型。

`util`包含了一些训练代码和评估代码。

`train` 是训练代码，可以直接运行相应的Train文件进行模型训练。

模型的训练过程中的训练日志保存在`log`文件中，每次迭代结束后保存模型文件到`checkpoint`文件夹中。


