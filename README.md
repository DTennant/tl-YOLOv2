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

3. Run the following commend to detect objects in an image, you can change the input image if you want to:
```bash
python3 main.py --run_mode image --input_img data/before.jpg --output_img data/after.jpg -
-model_path pretrained/tl-yolov2.ckpt
```

For Video detection, run the following commend:
```bash
python3 main.py --run_mode video --input_video data/source.avi --output_video data/target.avi --model_path pretrained/tl-yolov2.ckpt
```

You can checkout the detailed commend-line options by type in `python3 main.py -h`, the output:
```
usage: main.py [-h] [--run_mode {video,image}] [--input_video INPUT_VIDEO]
               [--output_video OUTPUT_VIDEO] [--input_img INPUT_IMG]
               [--output_img OUTPUT_IMG] [--model_path MODEL_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --run_mode {video,image}
                        the mode will be video or image
  --input_video INPUT_VIDEO
                        the video to be processed, default: capture from the
                        Camera(if available)
  --output_video OUTPUT_VIDEO
                        the path to save the video, default: play on the
                        screen(not saving)
  --input_img INPUT_IMG
                        the image to be processed
  --output_img OUTPUT_IMG
                        the path to save the output img
  --model_path MODEL_PATH
                        the path to the model ckpt
```

4. Result:

![before](https://raw.githubusercontent.com/DTennant/tl-YOLOv2/master/data/before.jpg)
![after](https://raw.githubusercontent.com/DTennant/tl-YOLOv2/master/data/after.jpg)

## TODO:

- Upload a video result
- Add Training phase, See [this issue](https://github.com/tensorlayer/tensorlayer/issues/435)
- Change PassThroughLayer and MyConcatLayer into proper tensorlayer layers