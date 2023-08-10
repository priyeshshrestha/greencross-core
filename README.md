# greencross-core
Core module of Greencross project for tiger detection

This application uses YoloV5 to detect tigers in the wild

### Steps to run

- install `pytorch`
- Get yolov5 repo from `https://github.com/ultralytics/yolov5`
- run `pip install -r requirements.txt` inside yolov5 directory
- Direct `sys.path.insert` in main_video.py/main_image.py to the yolov5 directory
- Put appropriate `model_path` and `image_path`/`video_path`
- For detection in video run `python3 main_video.py`
- For detection in image run `python3 main_image.py`
