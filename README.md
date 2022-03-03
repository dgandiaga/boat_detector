# Boat Detection and Tracking on videos using YOLOv5 and CSRT tracking

This repository performs boat detection on videos using a pretrained **YOLOv5** (https://pytorch.org/hub/ultralytics_yolov5/) and **CSRT tracking** (https://docs.opencv.org/3.4/d2/da2/classcv_1_1TrackerCSRT.html).


![boat_detection](https://user-images.githubusercontent.com/26325749/156657844-a128e28d-38b5-484b-8384-c7d63fcfb314.gif)

# Usage

Models and videos are not included in the repo due to their size. You can download them from Drive:
* Models: https://drive.google.com/file/d/1yoeDywoeW-GQmaFP9lHp80rDp9yfknMW/view?usp=sharing
* Videos: https://drive.google.com/file/d/1GHzaKvuYHg0a1h9XNNKLGE7w8Dq5pAe9/view?usp=sharing

You should extract both folders in the root folder of the project. Folder structure should end as follows:

```
project
│   README.md
│   requirements.txt
│   docker-compose.yml
│
└───src
│   │   image_utils.py
│   │   main.py
│   │   tests.py
│   
└───models
|   |   yolov5x.pt
│
└───videos
│   │   Test-Task Sequence from Wörthersee.mp4
|
└───docker
    │   Dockerfile.process_video

```

Having the model is in fact not necessary, but I recommend it because otherwise it'll have to download it everytime you run the docker process.

Once everything is downloaded you can build and launch the process. Everything is dockerized, so it can be built through the next command line:

    docker-compose build process_video
    
This command downloads the image, installs the requirements and runs some tests I made during development (you can check them in src/tests.py). Now you can run the docker process through the follow command:

    docker-compose run process_video
    
This outputs the result video in videos/result.mp4.

# Development

This repository uses a combination of detection using **YOLOv5** and tracking using **CSRT**. The YOLOv5 used for detection has a **threshold of 0.65** in order to avoid false positives. The side effect of this is that there are many false negatives, and detection itself is not enough for tracking the objects. The system keeps on memory the last position of an object for some time and if it reappears later and the bounding box overlaps with the last known position it is still able of knowing that this object was already counted. But sometimes the time without redetections is so long that when it reappears it's so far that the system is not able to link it with the previously seen boat. Here you can see what happens when the boat in the foreground is partially occluded. That's enough for spoiling detection for some seconds with this threshold, and when it reappears it's not able to match it with the last detection. You can see in green when the position is getting updated by the detector and in red when it's not detecting it and just keeping the last know position. By the time it gets detected again the system is not able to match it with the last known position.

![boat_only_detection](https://user-images.githubusercontent.com/26325749/156660010-199e1e45-221e-42cd-8b5f-aba34dabeba3.gif)

On the other side, **CSRT** is a tracking algorithm that is able to track a bounding box regardless of what it contains. I's very robust against occlusions, rotations or translations, but the downside is that trackers tend to accumulate errors because they try to follow all the bounding box and not only the boat, and sometimes they get stuck tracking a piece of background. Here I have an example using only tracking, once every boat is detected I apply a mask on the image so the detector doesn't return this boat again and the CSRT tracker takes the lead until it loses the bounding box.

![boat_tranking_only](https://user-images.githubusercontent.com/26325749/156663507-7daf0b81-92ff-4e61-8a31-3ddd1660bf9b.gif)


You can see that the tracker is able of keeping the position the boat, but it accumulates errors and finally the boat gets out of the image and the bounding box is still stuck tracking part of the background. You can also see on the second boat that in its first detection the bounding box is too big and the tracker only focuses on the background, so when the boat gets out of that area it gets detected again as a new boat.

The solution is a combination of both approaches. The final version of the project uses YOLOv5 detection when it's available for updating the bounding boxes and when it looses an object it switches to CSRT tracking, so when it appears again the system is able to match it with the previous detection while the detector's feedback keeps the tracker under control, so it doesn't go wild. Here is the same clip with this version of the project, in green when it's getting detected by the YOLO and in red when it's being followed by the CSRT.

![boat_detection_first](https://user-images.githubusercontent.com/26325749/156663011-bf2f0147-cc58-45da-8381-8e94535070fe.gif)

The solution is quite robust, and it's able to follow all the boats without false positives or duplicated detections, but there's still some margin of improvement on boats that are small or far from the camera and the detector is not able to get them due to the high confidence threshold imposed for avoiding false positives.

# Future Work

* **Trying more models**: I started with **yolov5s** and then added GPU support and with the performance improvement I switched to **yolov5x**.
* **Train the models**: Both models I used were pretrained on **COCO dataset** (https://cocodataset.org/#home). This dataset for object detection has around 90 classes, being "boat" among them. Maybe a model designed and trained for this specific use case gives better results.
* **Optimizing performance**: On Nvidia GeForce GTX 1650 this takes around 6 minutes for a 3-minute video. This is framed as an offline problem, but for online applications the performance should be improved. An alternative is switching from Python to C++. CSRT is implemented in OpenCV, so it's available in many languages, and YOLOv5 can ve exported to ONNX and other formats for model compression.
* **Reducing the threshold and applying post-processing for false positives**: This is feasible in order to be able to catch the smallest boats. For filtering false positives a simple option is imposing that many successful detections are required in a given time-window in order to consider it as true. There are also more sophisticated alternatives.
* **Using semantic segmentation for separating the foreground and improving detection**: I did several experiments with **fcn_resnet101** trained on COCO dataset, but it didn't lead to positive results, though I didn't invest much time.
* **Applying post-processing when trackers get stuck in the edge of the video following a piece of background as the boat leaves the image**: This still happens, even in the final version. The trackers are maintained alive during five seconds without detections until they are killed, and sometimes they get stuck in the edge of the video as the boat goes away. Solving this is not necessary for this particular use case because the goal is counting different boats, and keeping the tracker alive and stuck for many seconds even there's no boat won't hurt the count unless a new boat appears in that exact spot in the next five seconds, and that never happens. Still, in order to make the algorithm more robust this should be improved.

