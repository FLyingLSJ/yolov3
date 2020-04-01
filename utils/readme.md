# parse_config.py

### def parse_model_cfg(path):

```python
def parse_model_cfg(path):
    # Parse the yolo *.cfg file and return module definitions path may be 'cfg/yolov3.cfg', 'yolov3.cfg', or 'yolov3'
    """
    对 .cfg 文件进行解析
    得到的是一个列表，列表里面的值是字典，包含每个模块的具体参数
    """
    if not path.endswith('.cfg'):  # add .cfg suffix if omitted
        path += '.cfg'
    if not os.path.exists(path) and os.path.exists('cfg' + os.sep + path):  # add cfg/ prefix if omitted
        path = 'cfg' + os.sep + path

    with open(path, 'r') as f:
        lines = f.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    mdefs = []  # module definitions
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            mdefs.append({})
            mdefs[-1]['type'] = line[1:-1].rstrip()
            if mdefs[-1]['type'] == 'convolutional':
                mdefs[-1]['batch_normalize'] = 0  # pre-populate with zeros (may be overwritten later)
        else:
            key, val = line.split("=")
            key = key.rstrip()

            if key == 'anchors':  # return nparray
                mdefs[-1][key] = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))  # np anchors
            elif key in ['from', 'layers', 'mask']:  # return array
                mdefs[-1][key] = [int(x) for x in val.split(',')]
            else:
                val = val.strip()
                if val.isnumeric():  # return int or float
                    mdefs[-1][key] = int(val) if (int(val) - float(val)) == 0 else float(val)
                else:
                    mdefs[-1][key] = val  # return string

    # Check all fields are supported
    supported = ['type', 'batch_normalize', 'filters', 'size', 'stride', 'pad', 'activation', 'layers', 'groups',
                 'from', 'mask', 'anchors', 'classes', 'num', 'jitter', 'ignore_thresh', 'truth_thresh', 'random',
                 'stride_x', 'stride_y', 'weights_type', 'weights_normalization', 'scale_x_y', 'beta_nms', 'nms_kind',
                 'iou_loss', 'iou_normalizer', 'cls_normalizer', 'iou_thresh',
                 'ratio', 'reduction', 'kernelsize'] # 注意力机制

    f = []  # fields
    for x in mdefs[1:]:
        [f.append(k) for k in x if k not in f]
    u = [x for x in f if x not in supported]  # unsupported fields
    assert not any(u), "Unsupported fields %s in %s. See https://github.com/ultralytics/yolov3/issues/631" % (u, path)

    return mdefs
```

对 `yolo-tiny` 解析结果如下：

```bash
[{'type': 'net', 'batch': 64, 'subdivisions': 2, 'width': 416, 'height': 416, 'c
hannels': 3, 'momentum': '0.9', 'decay': '0.0005', 'angle': 0, 'saturation': '1.
5', 'exposure': '1.5', 'hue': '.1', 'learning_rate': '0.001', 'burn_in': 1000, '
max_batches': 500200, 'policy': 'steps', 'steps': '400000,450000', 'scales': '.1
,.1'}, 
{'type': 'convolutional', 'batch_normalize': 1, 'filters': 16, 'size': 3,
 'stride': 1, 'pad': 1, 'activation': 'leaky'}, 
{'type': 'maxpool', 'size': 2, 'stride': 2}, 
{'type': 'convolutional', 'batch_normalize': 1, 'filters': 32, 'siz
e': 3, 'stride': 1, 'pad': 1, 'activation': 'leaky'}, 
{'type': 'maxpool', 'size'
: 2, 'stride': 2}, {'type': 'convolutional', 'batch_normalize': 1, 'filters': 64
, 'size': 3, 'stride': 1, 'pad': 1, 'activation': 'leaky'}, 
{'type': 'maxpool',
'size': 2, 'stride': 2}, 
{'type': 'convolutional', 'batch_normalize': 1, 'filter
s': 128, 'size': 3, 'stride': 1, 'pad': 1, 'activation': 'leaky'}, {'type': 'max
pool', 'size': 2, 'stride': 2},
{'type': 'convolutional', 'batch_normalize': 1, 'filters': 256, 'size': 3, 'stride': 1, 'pad': 1, 'activation': 'leaky'}, 
{'type': 'maxpool', 'size': 2, 'stride': 2}, {'type': 'convolutional', 'batch_normalize': 1, 'filters': 512, 'size': 3, 'stride': 1, 'pad': 1, 'activation': 'leaky'},
{'type': 'maxpool', 'size': 2, 'stride': 1}, 
{'type': 'convolutional', 'batch_normalize': 1, 'filters': 1024, 'size': 3, 'stride': 1, 'pad': 1, 'activation': 'leaky'}, 
{'type': 'convolutional', 'batch_normalize': 1, 'filters': 256, 'size': 1, 'stride': 1, 'pad': 1, 'activation': 'leaky'}, 
{'type': 'convolutional', 'batch_normalize': 1, 'filters': 512, 'size': 3, 'stride': 1, 'pad': 1, 'activation': 'leaky'}, 
{'type': 'convolutional', 'batch_normalize': 0, 'size': 1, 'stride': 1, 'pad': 1, 'filters': 75, 'activation': 'linear'}, 
{'type': 'yolo', 'mask':[3, 4, 5], 'anchors': array([[ 10.,  14.],
       [ 23.,  27.],
       [ 37.,  58.],
       [ 81.,  82.],
       [135., 169.],
       [344., 319.]]), 'classes': 20, 'num': 6, 'jitter': '.3', 'ignore_thresh':
 '.7', 'truth_thresh': 1, 'random': 1}, 
{'type': 'route', 'layers': [-4]}, 
{'type': 'convolutional', 'batch_normalize': 1, 'filters': 128, 'size': 1, 'stride':
1, 'pad': 1, 'activation': 'leaky'}, 
{'type': 'upsample', 'stride': 2}, 
{'type': 'route', 'layers': [-1, 8]}, 
{'type': 'convolutional', 'batch_normalize': 1, 'filters': 256, 'size': 3, 'stride': 1, 'pad': 1, 'activation': 'leaky'}, 
{'type': 'convolutional', 'batch_normalize': 0, 'size': 1, 'stride': 1, 'pad': 1, 'filters': 75, 'activation': 'linear'}, 
{'type': 'yolo', 'mask': [0, 1, 2], 'anchors': array([[ 10.,  14.],
       [ 23.,  27.],
       [ 37.,  58.],
       [ 81.,  82.],
       [135., 169.],
       [344., 319.]]), 'classes': 20, 'num': 6, 'jitter': '.3', 'ignore_thresh':
 '.7', 'truth_thresh': 1, 'random': 1}]
```

