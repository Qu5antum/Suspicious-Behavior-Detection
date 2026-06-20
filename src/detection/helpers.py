import math

# Geometri yardımcıları
def bbox_center(box: list) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2

def bbox_iou(a: list, b: list) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)

def bbox_dist(a: list, b: list) -> float:
    cx1, cy1 = bbox_center(a)
    cx2, cy2 = bbox_center(b)
    return math.hypot(cx1 - cx2, cy1 - cy2)


def expand_bbox(box: list, px: int, fw: int, fh: int) -> list:
    x1, y1, x2, y2 = box
    return [max(0, x1 - px), max(0, y1 - px),
            min(fw, x2 + px), min(fh, y2 + px)]