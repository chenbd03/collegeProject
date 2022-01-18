# Style_transfer+GUI

## Introduction

This repository contains a pytorch implementation of an algorithm for artistic style transfer. The algorithm can be used to mix the content of an image with the style of another image.For example, this is a picture of dancing in the style of a mosaic glass decoration

![2421606149457_.pic_hd](https://billie-s-album.oss-cn-beijing.aliyuncs.com/img/2421606149457_.pic_hd.jpg)



## Environments

Python 3.6 and all requirements.txt dependencies installed, including torch==1.4.It is recommended to install the virtual environment here.The steps are as follows:

1、Install  Anaconda3

2、Create virtual environment，`conda create -n pytorch_p36 python=3.6`

3、activate virtual environment， `activate pytorch_p36`

4、Install dependencies

```
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install PyQt5==5.15.1
```



## Run the style_transfer

Run the GUI program 

```bash
python view.py
```

result：

![2441606149485_.pic_hd](https://billie-s-album.oss-cn-beijing.aliyuncs.com/img/2441606149485_.pic_hd.jpg)
