# Face_Detection_with_OpenCV
This project implements real-time face detection using a pre-trained Caffe deep learning model with OpenCV. It captures video from the webcam, detects faces, and highlights them with bounding boxes along with confidence scores.
Features:
Uses OpenCV's DNN module for face detection.
Highlights detected faces with bounding boxes in real-time.
Displays confidence scores for each detection.
Setup Instructions:
Clone the Repository:


git clone https://github.com/your-username/Face_Detection_with_OpenCV.git
cd Face_Detection_with_OpenCV
Install Dependencies: Ensure you have Python 3.6+ installed. Install required libraries:


pip install opencv-python numpy
Download Required Files:

Add deploy.prototxt and res10_300x300_ssd_iter_140000.caffemodel to the repository.
These files are available from OpenCV GitHub.
Usage:
Run the script:

python face_detection.py
Press q to exit the webcam feed.

File Descriptions:
face_detection.py: The main script for detecting faces using OpenCV.
deploy.prototxt: Defines the model's architecture.
res10_300x300_ssd_iter_140000.caffemodel: Pre-trained weights for the model.
Example Output:
Add a screenshot or GIF of the detection in action.

Dependencies:
Python 3.6+
OpenCV
NumPy


