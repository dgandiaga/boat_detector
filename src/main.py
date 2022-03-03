import cv2
import torch
import tqdm
from image_utils import get_tracker_bbox, get_detection_bbox, update_detections, get_closest_iou

DETECTION_LIFETIME = 2880 / 120 * 5  # Five seconds
IOU_THRESHOLD = 0  # As long as bboxes have a single pixel in common, they are labeled as the same object

MODEL_NAME = 'yolov5x'
VIDEO_NAME = 'Test-Task Sequence from Wörthersee.mp4'


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    # Initializing model, detections, progress bar and output buffer
    model = torch.hub.load('ultralytics/yolov5', MODEL_NAME, pretrained=True).to(device)
    current_detections = []
    number_of_detections = 0
    pbar = tqdm.tqdm(total=2880, position=0, leave=True)
    img_array = []

    # Opening video capture
    video = cv2.VideoCapture('videos/' + VIDEO_NAME)

    while video.isOpened():

        # Frame captioning
        ret, frame = video.read()

        # In case there's a problem with the video caption or the video is finished, stop
        if not ret:
            print(f'Error while loading video {VIDEO_NAME}')
            break

        else:
            # Saving video parameters
            height, width, layers = frame.shape
            size = (width, height)

        # Frame color conversion
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Update currently tracked objects
        detection_removal_indexes = []
        for i in range(len(current_detections)):

            # Remove a detection if it has exceeded its lifetime with no recent detections
            if current_detections[i][3] > DETECTION_LIFETIME:
                detection_removal_indexes.append(i)

            else:
                # Update detection living time for every tracked detection
                current_detections[i][3] += 1

                # Update tracker with the new frame
                ok, new_bbox = current_detections[i][4].update(frame)

                # Update the detection bbox with the tracker's feedback
                if ok:
                    detection_bbox = get_detection_bbox(new_bbox)
                    current_detections[i][1] = detection_bbox

                # If the tracker has failed, remove the detection
                else:
                    detection_removal_indexes.append(i)

        # Remove corrupt detections from memory
        current_detections = update_detections(current_detections, detection_removal_indexes)

        # Get new detections
        results = model(frame_rgb)
        results_df = results.pandas().xyxy[0]

        # Process and filter them
        results_df['size'] = (results_df['xmax'] - results_df['xmin']) * (results_df['ymax'] - results_df['ymin'])
        results_df = results_df[(results_df['class'] == 8)
                                & (results_df['confidence'] > 0.65)
                                & (results_df['size'] < 20000)]

        # Process every detection
        for xmin, ymin, xmax, ymax in results_df[['xmin', 'ymin', 'xmax', 'ymax']].values:

            # Extract detection information
            center = (int(xmin + (xmax - xmin) / 2), int(ymin + (ymax - ymin) / 2))
            bbox = (int(xmin), int(ymin), int(xmax), int(ymax))

            # If there are no detections, create the new one on memory with its data and tracker
            if len(current_detections) == 0:
                number_of_detections += 1
                current_detections.append([number_of_detections, bbox, center, 0, cv2.TrackerCSRT_create()])

                # Initialize tracker
                current_detections[-1][4].init(frame, get_tracker_bbox(bbox))

            # If there are detections, check if the detection is new or old
            else:

                # Get IoU for the closest detection on memory
                iou, closest_detection = get_closest_iou(center, bbox, current_detections)

                # Update that detection if it's considered as the same object
                if iou > IOU_THRESHOLD:

                    # Update detection bbox, center, frames since last detection and reinitialize its tracker
                    current_detections[closest_detection][1] = bbox
                    current_detections[closest_detection][2] = center
                    current_detections[closest_detection][3] = 0
                    current_detections[closest_detection][4] = cv2.TrackerCSRT_create()
                    current_detections[closest_detection][4].init(frame, get_tracker_bbox(bbox))

                else:

                    # Add new detection and initialize its tracker
                    number_of_detections += 1
                    current_detections.append([number_of_detections, bbox, center, 0, cv2.TrackerCSRT_create()])
                    current_detections[-1][4].init(frame, get_tracker_bbox(bbox))

        # Paint detections and ids on the output frame
        for detection in current_detections:

            # Define color regarding if the source is the detector or the tracker
            if detection[3] == 0:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            # Paint rectangle and text
            frame = cv2.rectangle(frame, (detection[1][0], detection[1][1]),
                                  (detection[1][2], detection[1][3]), color, 2)

            frame = cv2.putText(frame, f'Id: {detection[0]}', (detection[1][0], detection[1][1] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0))

        # Write detection data on the corner of the video
        frame = cv2.putText(frame, f'Detections: {number_of_detections}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255))
        frame = cv2.putText(frame, f'Tracked Boats: {len(current_detections)}', (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255))

        # Add the final frame to the output buffer and increase the progress bar
        img_array.append(frame)
        pbar.update(1)

    # Print summary and release input source
    print(f'Number of number_of_detections: {number_of_detections}')
    video.release()

    # Save output video to disk
    print('Writing...')
    result_video = cv2.VideoWriter('videos/result.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 15, size)

    for frame in img_array:
        result_video.write(frame)

    result_video.release()

    print('Finished')


if __name__ == "__main__":
    main()
