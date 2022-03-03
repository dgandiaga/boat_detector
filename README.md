# Boat Detection and Tracking on videos using YOLOv5 and CSRT tracking

This repository shows a pret

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

Once everything is downloaded you can buld and launch the process:

    docker-compose build process_video
    docker-compose run process_video
