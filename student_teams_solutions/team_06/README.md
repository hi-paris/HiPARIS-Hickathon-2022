## Start running object detection with YoloV5 in terminal specifying your image path

```
python detect.py --weights models/yolov5l/yolov5l.pt --img 250 --save-txt --save-conf --save-crop --conf 0.25 --source YOUR_IMAGE_PATH -name test
```
## Then run the script main.py 