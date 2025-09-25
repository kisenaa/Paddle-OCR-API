import os
import cv2
import copy
from . import predict_det
from . import predict_cls
from . import predict_rec
from .utils import get_rotate_crop_image, get_minarea_rect_crop


class TextSystem(object):
    def __init__(self, args):
        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

        self.args = args
        self.crop_image_res_index = 0

    def draw_crop_rec_res(self, output_dir, img_crop_list, rec_res):
        os.makedirs(output_dir, exist_ok=True)
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite(
                os.path.join(
                    output_dir, f"mg_crop_{bno+self.crop_image_res_index}.jpg"
                ),
                img_crop_list[bno],
            )

        self.crop_image_res_index += bbox_num

    def __call__(self, img, cls=True):
        import os
        import numpy as np

        ori_im = img.copy()
        # text detection
        dt_boxes = self.text_detector(img)

        # If nothing detected return empty results in consistent format
        if dt_boxes is None or len(dt_boxes) == 0:
            return np.array([]), []

        img_crop_list = []
        dt_boxes = sorted_boxes(dt_boxes)

        # crop detected boxes
        for bno in range(len(dt_boxes)):
            box = dt_boxes[bno].astype("int32")
            # get_rotate_crop_image expects the original image and box points
            try:
                tmp_crop = get_rotate_crop_image(ori_im, box)
            except Exception:
                # fallback: extract bounding rect
                rect = get_minarea_rect_crop(ori_im, box)
                tmp_crop = rect if rect is not None else ori_im
            img_crop_list.append(tmp_crop)

        # angle classifier (if enabled)
        if self.use_angle_cls and cls:
            img_crop_list, angle_list = self.text_classifier(img_crop_list)

        # recognition
        rec_res = []
        if len(img_crop_list) > 0:
            rec_res = self.text_recognizer(img_crop_list)
        else:
            rec_res = []

        # optionally save crops
        if getattr(self.args, "save_crop_res", False):
            out_dir = getattr(self.args, "save_crop_res_dir", "output")
            self.draw_crop_rec_res(out_dir, img_crop_list, rec_res)

        # filter low-score results and keep alignment between boxes and rec_res
        # Defensive: ensure rec_res is a list aligned with dt_boxes
        if rec_res is None:
            rec_res = []
        rec_res = list(rec_res)
        if len(rec_res) < len(dt_boxes):
            # pad missing recognition results with empty entries
            rec_res += [["", 0.0]] * (len(dt_boxes) - len(rec_res))

        filter_boxes = []
        filter_rec_res = []
        for box, rec in zip(dt_boxes, rec_res):
            # rec is expected to be [text, score] or similar
            score = 0.0
            try:
                score = float(rec[1])
            except Exception:
                score = 0.0
            if score >= getattr(self, "drop_score", 0.0):
                filter_boxes.append(box)
                filter_rec_res.append(rec)

        # Return filtered boxes and their recognition results (keep as list + ndarray)
        return np.array(filter_boxes), filter_rec_res


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (
                _boxes[j + 1][0][0] < _boxes[j][0][0]
            ):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes
