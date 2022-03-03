import unittest

from image_utils import bb_intersection_over_union, get_tracker_bbox, get_detection_bbox, update_detections, \
    get_closest_iou


class TestUtils(unittest.TestCase):

    def test_bb_intersection_over_union(self):
        bb1 = [2, 2, 2, 2]
        bb2 = [2, 2, 2, 2]

        iou = bb_intersection_over_union(bb1, bb2)
        self.assertEqual(iou, 1)

        bb1 = [2, 2, 2, 2]
        bb2 = [2, 2, 3, 3]

        iou = bb_intersection_over_union(bb1, bb2)
        self.assertEqual(iou, 1 / 4)

        bb1 = [1, 1, 2, 2]
        bb2 = [2, 2, 3, 3]

        iou = bb_intersection_over_union(bb1, bb2)
        self.assertEqual(iou, 1 / 7)

    def test_get_tracker_bbox(self):
        bb = (2, 2, 4, 6)

        bb_tracker = get_tracker_bbox(bb)
        self.assertEqual(bb_tracker, (2, 2, 2, 4))

    def test_get_detection_bbox(self):
        bb = (2, 2, 2, 4)

        bb_tracker = get_detection_bbox(bb)
        self.assertEqual(bb_tracker, (2, 2, 4, 6))

    def test_get_closest_iou(self):
        center = (10, 10)
        bbox = (7, 7, 13, 13)
        current_detections = ((0, (8, 8, 8, 8), (11, 11), None, None),
                              (1, (8, 8, 12, 12), (11, 10), None, None),
                              (2, (8, 8, 16, 16), (13, 13), None, None))

        iou, detection_index = get_closest_iou(center, bbox, current_detections)

        self.assertEqual(iou, 5*5 / (7*7))
        self.assertEqual(detection_index, 1)

    def test_update_detections(self):
        detections = [1, 2, 3, 4, 5]
        indexes = [0, 3]

        updated_detections = update_detections(detections, indexes)

        self.assertEqual(updated_detections, [2, 3, 5])


if __name__ == '__main__':
    unittest.main()
