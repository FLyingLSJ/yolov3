本项目 Fork 自  https://github.com/ultralytics/yolov3 感谢大佬开源！！！

[官方文档](README_OFFICIAL.md)

### 环境准备

```bash
pip install -U -r requirements.txt
```

以下以训练 VOC2007 数据集为例，tiny 预训练模型 

VOC数据文件结构如下

```bash
- VOC2007
	- Annotations # XML文件
	- ImageSets 
	- Main
		|- test.txt
	    |- train.txt
    	|- trainval.txt
    	|- val.txt
	- JPEGImages # 图片
```

​	我们将 VOC 数据集下载下来放在项目的 data 文件夹中

```bash
cd data # 进入本项目的 data 文件夹
ln -s VOC2007/JPEGImages ./images # 建立软连接
```

~~mkdir images~~

~~ln -s /input0/VOC2007/JPEGImages ./home/yolov3/data/images/~~ 

### 创建 data/custom.names 文件

```bash
aeroplane
bicycle
bird
boat
bottle
bus
car
cat
chair
cow
diningtable
dog
horse
motorbike
person
pottedplant
sheep
sofa
train
tvmonitor
```

### 更新 data/custom.data, 其中保存的是配置信息

```bash
classes= 20
train=data/train_voc_2007.txt # 这个放在 VOC.zip 文件下了，解压后移动到 data 下面即可
valid=data/val_voc_2007.txt
names=data/custom.names
```

### cfg 文件配置

`cfg/yolov3-tiny-20cls.cfg`

训练`Training`和测试 `Testing`需要修改一下 

```bash
[net]
# Testing
#batch=1
#subdivisions=1
# Training
batch=64
subdivisions=2
width=416
height=416
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.001
burn_in=1000
max_batches = 500200
policy=steps
steps=400000,450000
scales=.1,.1

[convolutional]
batch_normalize=1
filters=16
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=1

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

###########

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=75
activation=linear



[yolo]
mask = 3,4,5
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
classes=20
num=6
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1

[route]
layers = -4

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[upsample]
stride=2

[route]
layers = -1, 8

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=75
activation=linear

[yolo]
mask = 0,1,2
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
classes=20
num=6
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1
```

### 下载权重
tiny 网络 
```bash
cd weights
wget -c https://pjreddie.com/media/files/yolov3-tiny.weights
```

### 训练

```bash
python train.py --data data/custom.data --cfg cfg/yolov3-tiny-20cls.cfg --weights weights/yolov3-tiny.weights --epoch 50 --batch 16 --accum 4 --multi
```

---

```bash
Epoch  gpu_mem   GIoU  obj   cls   total  targets img_size
9/272    7.08G   4.05  1.84  4.34  10.2   29      416: 100%|█████| 157/157 [01:37<00:00,  1.60it/s]
   Class  Images    Targets   P     R      mAP@0.5   F1: 100%|█████| 79/79 [00:28<00:00,  2.73it/s]
   all    2.51e+03  6.31e+03  0.39  0.674  0.429     0.487
```


### 测试（对 data/samples 文件夹里面的图片进行检测，结果在 output 文件夹下）

```bash
python detect.py --cfg cfg/yolov3-tiny-20cls.cfg --names data/custom.names --weights weights/best.pt
```

### 模型评估

```bash
python test.py --cfg cfg/yolov3-tiny-20cls.cfg --data data/custom.data --weights weights/best.pt
```
---
```bash
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%
                 all  2.51e+03  6.31e+03    0.0207     0.852     0.534    0.0404
           aeroplane  2.51e+03       155    0.0349     0.903     0.681    0.0672
             bicycle  2.51e+03       177    0.0158     0.921     0.324     0.031
                bird  2.51e+03       243    0.0182     0.761     0.525    0.0355
                boat  2.51e+03       150    0.0184     0.753     0.466    0.0359
              bottle  2.51e+03       252    0.0131     0.504     0.146    0.0256
                 bus  2.51e+03       114     0.018     0.912     0.726    0.0353
                 car  2.51e+03       625    0.0241     0.789     0.608    0.0467
                 cat  2.51e+03       190    0.0379     0.926     0.752    0.0728
               chair  2.51e+03       398    0.0204     0.847     0.408    0.0398
                 cow  2.51e+03       123    0.0169     0.943       0.5    0.0332
         diningtable  2.51e+03       112    0.0219     0.902     0.565    0.0428
                 dog  2.51e+03       257    0.0233     0.946     0.646    0.0455
               horse  2.51e+03       180    0.0183     0.983     0.799    0.0359
           motorbike  2.51e+03       172    0.0179     0.942     0.348    0.0351
              person  2.51e+03  2.33e+03    0.0186     0.916     0.518    0.0365
         pottedplant  2.51e+03       266    0.0138     0.602     0.202    0.0269
               sheep  2.51e+03       127    0.0131     0.717      0.46    0.0258
                sofa  2.51e+03       124    0.0188     0.935     0.473    0.0368
               train  2.51e+03       152    0.0381     0.974     0.894    0.0733
           tvmonitor  2.51e+03       158    0.0136     0.873     0.642    0.0268

```

### 绘制训练曲线

```bash
python plot_results.py 
```

![](results.png)

### 环境相关

以下的环境测试成功过

```bash
numpy                  1.18.1                   
opencv-python          4.1.1.100          
pandas                 1.0.0                  
Pillow                 6.2.2            
scikit-image           0.16.2         
scikit-learn           0.22.1         
scipy                  1.4.1                
tensorboard            1.14.0         
tensorflow             1.14.0.100     
tensorflow-estimator   1.14.0         
tensorflow-gpu         1.14.0         
tensorflow-serving-api 1.14.0              
torch                  1.2.0          
torchvision            0.4.0             
tqdm                   4.42.1           
```

###  超参数选取

```bash
python train.py --data data/custom.data --cfg cfg/yolov3-tiny-20cls.cfg --img-size 416 --epochs 2 --evolve --weights weights/yolov3-tiny.weights 
```

需要修改 train.py 最后面 `for _ in range(1):  # generations to evolve`

-  https://github.com/ultralytics/yolov3/issues/392 
-  https://mp.weixin.qq.com/s?__biz=MzA4MjY4NTk0NQ==&mid=2247484757&idx=2&sn=abd254591a6a46077141e2356159c37d&chksm=9f80bfc3a8f736d50156f7b2939967587f5ddb85eb4ec7c88bdb6e0cc9aad2b00ebbc0888ecf&scene=21#wechat_redirect 