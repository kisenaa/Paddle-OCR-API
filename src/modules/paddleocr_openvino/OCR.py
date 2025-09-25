import time

from .predict_system import TextSystem
from .utils import infer_args as init_args
from .utils import str2bool, draw_ocr
import argparse
import sys


class OCR(TextSystem):
    def __init__(self, **kwargs):
        # 默认参数
        parser = init_args()
        inference_args_dict = {}
        for action in parser._actions:
            inference_args_dict[action.dest] = action.default
        params = argparse.Namespace(**inference_args_dict)

        # params.rec_image_shape = "3, 32, 320"
        params.rec_image_shape = "3, 48, 320"

        # 根据传入的参数覆盖更新默认参数
        params.__dict__.update(**kwargs)

        # 初始化模型
        super().__init__(params)

    def ocr(self, img, det=True, rec=True, cls=False):
        if cls == True and self.use_angle_cls == False:
            print(
                "Since the angle classifier is not initialized, the angle classifier will not be uesd during the forward process"
            )

        if det and rec:
            ocr_res = []
            dt_boxes, rec_res = self.__call__(img, cls)
            tmp_res = [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
            ocr_res.append(tmp_res)
            return ocr_res
        elif det and not rec:
            ocr_res = []
            dt_boxes = self.text_detector(img)
            tmp_res = [box.tolist() for box in dt_boxes]
            ocr_res.append(tmp_res)
            return ocr_res
        else:
            ocr_res = []
            cls_res = []

            if not isinstance(img, list):
                img = [img]
            if self.use_angle_cls and cls:
                img, cls_res_tmp = self.text_classifier(img)
                if not rec:
                    cls_res.append(cls_res_tmp)
            rec_res = self.text_recognizer(img)
            ocr_res.append(rec_res)

            if not rec:
                return cls_res
            return ocr_res


def sav2Img(org_img, result, name="draw_ocr.jpg"):
    import cv2
    import base64
    import numpy as np
    from PIL import Image as PILImage

    if not result:
        return None

    # unwrap nested result
    res = result[0] if isinstance(result, (list, tuple)) and len(result) > 0 else result

    # convert original BGR (cv2.imread) to RGB for drawing if needed
    image_rgb = org_img[:, :, ::-1]

    boxes = [line[0] for line in res]
    txts = [line[1][0] for line in res]
    scores = [line[1][1] for line in res]
    im_show = draw_ocr(image_rgb, boxes, txts, scores)

    # normalize draw_ocr output to a PIL RGB image
    if hasattr(im_show, "convert"): 
        pil_rgb = im_show.convert("RGB")
    elif isinstance(im_show, np.ndarray):
        pil_rgb = PILImage.fromarray(im_show.astype(np.uint8)).convert("RGB")
    else:
        pil_rgb = PILImage.fromarray(np.array(im_show)).convert("RGB")

    # get numpy RGB and convert to BGR for cv2
    im_rgb = np.array(pil_rgb)
    im_bgr = im_rgb[:, :, ::-1]

    # save + encode
    # try:
    #     cv2.imwrite(name, im_bgr)
    # except Exception:
    #     pass

    ok, buf = cv2.imencode(".jpg", im_bgr)
    if not ok:
        return None

    b64_img = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_img}"

if __name__ == "__main__":
    import cv2

    model = OCR(use_angle_cls=False, use_gpu=False)

    img = cv2.imread("test_image.jpg")
    s = time.time()
    result = model.ocr(img)
    e = time.time()
    print("total time: {:.3f}".format(e - s))
    print("result:", result)
    for box in result[0]:
        print(box)

    sav2Img(img, result)
