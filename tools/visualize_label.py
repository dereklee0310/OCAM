import cv2
import numpy as np

image = cv2.imread("./datasets/fold02/train/images/0043.png")

with open("./datasets/fold02/train/labels/0057.txt", "r") as label_file:
    for line in label_file:
        line = map(float, line[2:].split())
        line = [int(x * 500) if i % 2 == 0 else int(x * 1200) for i, x in enumerate(line)]
        line = [line[i:i+2] for i in range(0,len(line),2)]
        print(line)
        cv2.polylines(image, [np.array(line).astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=4)

    cv2.imwrite("gt.png", image)
    cv2.waitKey(0)