import yaml
import math
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import json
import cv2

CONFIG = "configs/inference.yaml"
IMAGE = "datasets/fold01/val/images/0041.png"

def get_max_angle_diff(angles, xywhr):
    idx1 = idx2 = -1
    max_angle_diff = -1
    for i in range(len(angles)):
        for j in range(i, len(angles)):
            angle = get_radians_diff(angles[i], angles[j])
            if angle > max_angle_diff:
                max_angle_diff = angle
                idx1, idx2 = i, j
    # print("--------------------Content--------------------")
    # print("max:", radian2angle(max_angle_diff))
    # print("idx1:", idx1)
    # print("idx2:", idx2)
    # print("angle1:", radian2angle(xywhr[idx1][4])) # xywhr
    # print("angle2:", radian2angle(xywhr[idx2][4])) # xywhr
    return radian2angle(max_angle_diff), idx1, idx2


def get_radians_diff(angle1, angle2):
    abs_diff = abs(angle1 - angle2)
    return min(abs_diff, math.pi - abs_diff)  # cycle diff


def radian2angle(radian):
    # also convert large radian to negative value
    angle = radian * 180 / math.pi
    return angle if angle < 90 else angle - 180


def get_mse(pred, label):
    return np.mean((np.array(label) - np.array(pred)) ** 2)


def get_rmse(pred, label):
    return np.sqrt(get_mse(pred, label))

def draw_lines(image, pts):
    cv2.polylines(image, [np.array(pts)], isClosed=True, color=(255, 0, 0), thickness=4)


def draw_box_and_line(image, xyxyxyxy, upper):
    draw_lines(image, xyxyxyxy)
    xyxyxyxy.sort(key=lambda x:x[0])
    x1, y1 = xyxyxyxy[1]
    x2, y2 = xyxyxyxy[0]
    x2 = x2 + 1 if x1 == x2 else x2
    m, b = np.polyfit((x1, x2), (y1, y2), 1)
    x1, y1 = (0, int(b)) if b >= 0 else (int(-b // m), 0)
    x2, y2 = (500, int(m * 500 + b)) if m * 500 + b <= 1200 else (int((1200 - b) // m), 1200)
    draw_lines(image, [[x1, y1], [x2, y2]])

    xyxyxyxy.sort(key=lambda x:x[1])
    x1, y1 = xyxyxyxy[1] if upper else xyxyxyxy[3]
    x2, y2 = xyxyxyxy[0] if upper else xyxyxyxy[2]
    x2 = x2 + 1 if x1 == x2 else x2
    m, b = np.polyfit((x1, x2), (y1, y2), 1)
    x1, y1 = (0, int(b)) if b >= 0 else (int(-b // m), 0)
    x2, y2 = (500, int(m * 500 + b)) if m * 500 + b <= 1200 else (int((1200 - b) // m), 1200)
    draw_lines(image, [[x1, y1], [x2, y2]])

def main():
    try:
        with open(CONFIG, "r") as cfg_file:
            cfg = yaml.safe_load(cfg_file)
    except yaml.YAMLError as e:
        print(e)

    # quick override for test
    cfg["save"] = False
    cfg["model"] = "runs/l_mosaic0_v0_batch8_epoch200/fold01/train/weights/best.pt"
    cfg["source"] = IMAGE
    model = YOLO(cfg["model"])
    results = model.predict(**cfg)

    pred_angle_diffs = []
    pred_idx_pairs = []
    for result in results:
        xywhr = result.obb.xywhr.to("cpu").numpy().copy().tolist()
        # xywhr = [y for x, y in zip(result.obb.conf, xywhr) if x >= 0.8]
        xywhr.sort(key=lambda x: x[1])  # sort by y coord
        angles = [x[4] for x in xywhr]
        max_angle_diff, idx1, idx2 = get_max_angle_diff(angles, xywhr)
        pred_angle_diffs.append(max_angle_diff)
        pred_idx_pairs.append([idx1, idx2])

        result.save("test.png")
        image = cv2.imread(IMAGE)
        xyxyxyxy = result.obb.xyxyxyxy.to("cpu").numpy().copy().astype(np.int32).tolist()
        xyxyxyxy = sorted(xyxyxyxy, key=lambda x:x[0][1]) # sort by y of first (x, y)

        upper, lower = sorted([xyxyxyxy[idx1], xyxyxyxy[idx2]], key=lambda x: x[0][1])
        draw_box_and_line(image, upper, upper=True)
        draw_box_and_line(image, lower, upper=False)
        cv2.imwrite("test2.png", image)


    # assume each image has a angle label file :p
    labels = []
    label_idx_pairs = []
    # label_dir = Path(cfg["source"]).parents[0] / "angles"  # datasets/fold01/val/angles
    label_dir = Path("datasets/fold01/val/angles")
    for filename in sorted(label_dir.iterdir())[:len(results)]:
        with open(filename, "r") as file:
            data = json.load(file)
            labels.append(data["diff"])
            label_idx_pairs.append([data["min_idx"], data["max_idx"]])

    for pred, label in zip(pred_angle_diffs, labels):
        print(f"pred: {pred:5.2f} | label: {label:5.2f}")

    for pred, label in zip(pred_idx_pairs, label_idx_pairs):
        print(pred, label)

    print("mse:", get_mse(pred_angle_diffs, labels))
    print("rmse:", get_rmse(pred_angle_diffs, labels))



if __name__ == "__main__":
    main()
