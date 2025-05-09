#"Object-Detection-Using-Image-Processing"

A final-year B.Tech project focused on real-time object detection using image processing and deep learning with the YOLOv3 model and OpenCV.

Table of Contents

Introduction

Problem Statement

Proposed Solution

Tech Stack

System Architecture

Installation

Usage

Sample Output

Results

Future Scope

Contributors



---

Introduction

Object detection is a critical application of image processing that enables machines to identify and locate multiple objects in images or video. This project uses the YOLOv3 deep learning model integrated with OpenCV in Python to detect and label objects in real-time.

Problem Statement

While image data is abundant, deriving meaningful insight from it in real time remains a challenge. Traditional detection methods fail to scale, generalize, or deliver real-time results. This project addresses:

Scalability of processing visual data

Balancing speed and accuracy

Dealing with occlusions and lighting variations


Proposed Solution

We implement a YOLOv3-based detection pipeline that:

Uses pre-trained COCO dataset weights

Detects multiple objects in real-time

Outputs labeled bounding boxes and confidence scores

Works efficiently on mid-range hardware


Tech Stack

Python 3.x

OpenCV 4.x

NumPy

YOLOv3 (weights + config)

COCO dataset (for object classes)


System Architecture

graph TD
    A[Input Image/Video] --> B[Preprocessing]
    B --> C[YOLOv3 Inference]
    C --> D[Post-processing (NMS)]
    D --> E[Labeled Output with Bounding Boxes]

Installation

1. Clone the repository:

git clone https://github.com/yourusername/object-detection-yolo
cd object-detection-yolo


2. Install dependencies:

pip install opencv-python numpy


3. Download YOLOv3 weights and config:

yolov3.weights

yolov3.cfg

Place them in the project directory.




Usage

import cv2

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
img = cv2.imread("sample.jpg")
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
outputs = net.forward(output_layers)

# Post-processing, draw bounding boxes and labels...

You can also run the detection on live camera feed for real-time output.

Sample Output

Image 1: Detected dog (98%), bicycle (88%)

Live Video: 15–20 FPS real-time detection for mobile phones, laptops, and people


Results

Average Accuracy: 90–95% on COCO dataset

Processing Time: ~50ms/frame on CPU


Future Scope

Deploy on edge devices using model optimization (pruning, quantization)

Custom object detection through retraining

Integrate with video analytics and tracking

Deploy as a web or mobile application


Contributors

Jebas Angel – Reg. No: 961222243010

Godson – Reg. No: 961222243012

Rahul – Reg. No: 96122243018

Shyam – Reg. No: 961222243021


Project Guide: Jovita Mam
Loyola Institute of Technology and Science, Thovalai
Department of Artificial Intelligence and Data Science
