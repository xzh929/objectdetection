import numpy as np
from iou import IOU


def nms(boxes, thresh=0.3, isMin=False):
    if boxes.shape[0] == 0:
        return np.array([])
    _boxes = boxes[(-boxes[:, 4]).argsort()]

    r_boxes = []
    while _boxes.shape[0] > 1:
        a_box = _boxes[0]
        b_boxes = _boxes[1:]
        r_boxes.append(a_box)
        index = np.where(IOU(a_box, b_boxes, isMin) < thresh)
        _boxes = b_boxes[index]
    if _boxes.shape[0] > 0:
        r_boxes.append(_boxes[0])
    return np.stack(r_boxes)
