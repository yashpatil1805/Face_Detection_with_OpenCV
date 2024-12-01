import cv2
import numpy as np

# Paths to the pre-trained deep learning face detector model files
prototxt = "deploy.prototxt"  # Path to the .prototxt file
model = "res10_300x300_ssd_iter_140000.caffemodel"  # Path to the .caffemodel file

# Load the model
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Initialize the video capture
video = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Get the frame's dimensions
    (h, w) = frame.shape[:2]
    
    # Pre-process the frame to create a blob
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    # Pass the blob through the network to obtain face detections
    net.setInput(blob)
    detections = net.forward()
    
    # Loop over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence (probability) of the detection
        confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections with a confidence threshold
        if confidence > 0.5:
            # Compute the (x, y) coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Ensure the bounding box is within the frame dimensions
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w, endX), min(h, endY)
            
            # Draw the bounding box around the face
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            
            # Display the confidence value on the bounding box
            text = f"{confidence * 100:.2f}%"
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Face Detection', frame)
    
    # Break the loop when 'q' is pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
video.release()
cv2.destroyAllWindows()
