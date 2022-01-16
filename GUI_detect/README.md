
# Yolov5 + Deep Sort with PyTorch



## Prepare

​	1、Create a virtual environment with python >= 3.6 , please create virtual environment first.

```
conda activate [virtual environment name]
```

​	2、Install relative package, pytorch>=1.6.0, torchvision>=0.7.0, pyqt5 >=5.15.4.   In the project, the **requirements.txt** is my own environments. You can refer to it.
​		【note】Please distinguish individual CUDA versions

```
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

​	3、Install all dependencies

```
pip install -r requirements.txt
```

​	4、Download the yolov5.weight.

 Please go to [official site of yolov5](https://github.com/ultralytics/yolov5),  download the yolov5 weight, and put the weights to the path ./yolov5/weights/



## Run

```
python main.py
【note】This project only realizes the pedestrian and vehicle detection targets. Please refer to the code(ui_main.py) for details.
First, load the model, then select the image path or video path, or upload the RTSP data stream (0 can be entered here to call the local camera), and then click start detection. Eventually you'll see the results.
```


## Reference

​	1、code： [the official of yolov5](https://github.com/ultralytics/yolov5) 

​	2、code： [deep_sort_pytorch](https://github.com/ZQPei/deep_sort_pytorch)  、paper:([Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402)

