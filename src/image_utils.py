import numpy as np

# Computing IoU of two bounding boxes
def bb_intersection_over_union(boxA, boxB):
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of the intersection
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of the union
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


# Refactor detection bbox (x0, y0, xf, yf) to tracker bbox (x0, y0, h, w)
def get_tracker_bbox(tracker_bbox):
    detection_bbox = (int(tracker_bbox[0]), int(tracker_bbox[1]),
                    int(tracker_bbox[2] - tracker_bbox[0]), int(tracker_bbox[3] - tracker_bbox[1]))
    return detection_bbox


# Refactor tracker bbox (x0, y0, h, w) to detection bbox (x0, y0, xf, yf)
def get_detection_bbox(detection_bbox):
    xmin, ymin, xmax, ymax = detection_bbox
    tracker_bbox = (int(xmin), int(ymin), int(xmax + xmin), int(ymax + ymin))
    return tracker_bbox


# Remove corrupt detections
def update_detections(current_detections, detection_removal_indexes):
    for i in range(len(detection_removal_indexes)):
        # Get the item that has to be removed and delete it
        current_index = detection_removal_indexes[i]
        del current_detections[current_index]

        # Update remaining deletion indexes because now detection list is one item shorter
        detection_removal_indexes = [detection_index if detection_index <= current_index else detection_index - 1
                                     for detection_index in detection_removal_indexes]

    return current_detections


# Get the IoU of the closest detection on memory
def get_closest_iou(center, bbox, current_detections):
    # Calculate distances among centers of detections
    distances_to_detections = [np.linalg.norm(np.array(center) - np.array(detection[2]))
                               for detection in current_detections]

    # Get the closest detection
    closest_detection = np.argmin(distances_to_detections)

    # Get the IoU for that detection
    iou = bb_intersection_over_union(bbox, current_detections[closest_detection][1])

    return iou, closest_detection
