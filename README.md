# tl-YOLOv2
A tensorlayer implementation of [YOLOv2](http://pjreddie.com/darknet/yolo/)

## Environment

- python 3.6 or 3.5
- Anaconda 4.2.0
- tensorlayer 1.7.4
- tensorflow 1.4.1 or 1.4.0
- opencv-python 3.4.*

## Usage

1. Clone the repo and cd into it: `git clone https://github.com/dtennant/tl-YOLOv2.git && cd tl-YOLOv2/`

2. Download the weights pretrained on COCO dataset from [BaiduYun](https://pan.baidu.com/s/1t7FGZyEB88MF6fAaLCZOzw) and unzip it into `pretrained/`

3. Run the following commend to interference, you can change the input image if you want to:
```bash
python main.py --input_img data/before.jpg --model_path pretrained/tl-yolov2.ckpt --output_img data/after.jpg
```

4. Result:
![before](https://raw.githubusercontent.com/DTennant/tl-YOLOv2/master/data/before.jpg)
![after](https://raw.githubusercontent.com/DTennant/tl-YOLOv2/master/data/after.jpg)

## TODO:

- Add Training phase, See [this issue](https://github.com/tensorlayer/tensorlayer/issues/435)
- Change PassThroughLayer and MyConcatLayer into proper tensorlayer layers