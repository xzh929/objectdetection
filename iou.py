import numpy as np


def IOU(box, boxes, isMin=False):
    box_area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    lx = np.maximum(box[:, 0], boxes[:, 0])
    ly = np.maximum(box[:, 1], boxes[:, 1])
    rx = np.minimum(box[:, 2], boxes[:, 2])
    ry = np.minimum(box[:, 3], boxes[:, 3])

    w = np.maximum(0, rx - lx)
    h = np.maximum(0, ry - ly)

    inter = w * h
    if isMin:
        ovr = np.true_divide((inter, np.minimum(box_area, boxes_area)))
    else:
        ovr = np.true_divide(inter, (box_area + boxes_area - inter))

    return ovr


if __name__ == '__main__':
    a = np.array([[1, 1, 10, 10], [10, 10, 20, 20]])
    b = np.array([[2, 2, 8, 8], [40, 40, 50, 50]])
    print(IOU(a, b))
