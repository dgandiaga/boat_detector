# Boat Detection and Tracking on videos using YOLOv5 and CSRT tracking

This repository performs boat detection on videos using a pretrained YOLOv5 (https://pytorch.org/hub/ultralytics_yolov5/) and CSRT tracking (https://docs.opencv.org/3.4/d2/da2/classcv_1_1TrackerCSRT.html).


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

Having the model is in fact not neccessary, but I recommend it because otherwise it'll have to download it everytime you run the docker process.

Once everything is downloaded you can buld and launch the process. Everything is dockerized, so it can be built through the next command line:

    docker-compose build process_video
    
This command downloads the image, installs the requirements and runs some tests I made during development (you can check them in src/tests.py). Now you can run the docker process through the follow command:

    docker-compose run process_video
    
This outputs the result video in videos/result.mp4.

# Development

This repository uses a combination of detection using **YOLOv5** and tracking using **CSRT**. The YOLOv5 used for detection has a **threshold of 0.65** in order to avoid false positives. The side effect of this is that there are many false negatives, and detection itself is not enough for tracking the objects. The system keeps on memory the last position of an object for some time and if it reapears later and the bounding box overlaps with the last known position it is still able of knowing that this object was already counted. But sometimes the time without redetections is so long that when it reappears it's so far than the system is not able to link it with the previously seen boat. Here you can see what happens when the boat in the foreground is partially occluded. That's enough for spoiling detection for some seconds with this threshold, and when it reappears it's not able to match it with the last detection. Here you can see in green when the position is getting updated by the detector and in red when it's not detecting it and just keeping the last know position. By the time it gets detected again the system is not able to match it with the last known position.

![boat_only_detection](https://user-images.githubusercontent.com/26325749/156660010-199e1e45-221e-42cd-8b5f-aba34dabeba3.gif)

On the other side, **CSRT** is a tracking tool that is able to track a bounding box regardless what it contains. The downside of this is that trackers tend to accumulate errors becase they try to follow all the bounding box and not only the boat, and sometimes they get stucked tracking a piece of background. Here I have an example using only tracking, once every boat is detected I apply a mask on the image so the detector doesn't return this boat again and the CSRT tracker takes the lead until it loses the bounding box.

![boat_only_tracking](https://user-images.githubusercontent.com/26325749/156661276-6f7700ee-6e91-4a97-a46a-73748a9c739e.gif)

You can see that the tracker is able of keeping the boat, but it accumulates errors and finally the boat gets out of the image and the bounding box is still stucked tracking part of the background. Also you can see on the second boat that in its first detection the bounding box is too big and the tracker only focuses on the background, so when the boat gets out of it it gets detected again as a new boat.

The solution is a combination of both approaches. The final version of the project uses detection when it can, and when it looses an object it switches to tracking, so when it appears again the system is able to match it with the previous detection at the same time that the detector's feedback keeps the tracker uner control so it doesn't grow wild. Here it is the same clip with this version of the project, in green when it's getting detected by the YOLO and in red when it's being followed by the CSRT.

![detection_boats_first](https://user-images.githubusercontent.com/26325749/156662391-0986130e-1828-47ba-9b44-109a9e91c24d.gif)
