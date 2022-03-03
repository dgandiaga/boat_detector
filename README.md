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

This repository uses a combination of detection using **YOLOv5** and tracking using **CSRT**. The YOLOv5 used for detection has a **threshold of 0.65** in order to avoid false positives. The side effect of this is that there are many false negatives, and detection itself is not enough for tracking the objects. The system keeps on memory the last position of an object for some time and if it reapears later and the bounding box overlaps with the last known position it is still able of knowing that this object was already counted. But sometimes the time without redetections is so long that when it reappears it's so far than the system is not able to link it with the previously seen boat. Here you can see what happens when the boat in the foreground is partially occluded. That's enough for spoiling detection for some seconds with this threshold, and when it reappears it's not able to match it with the last detection:


