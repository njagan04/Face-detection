import time

import cv2

from face_detection.scrfd.detector import SCRFD
from face_detection.yolov5_face.detector import Yolov5Face

# Initialize the face detector
#detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5m-face.pt")
detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")


def main():
    pass


if __name__ == "__main__":
    main()
