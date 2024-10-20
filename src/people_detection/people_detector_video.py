import cv2
import numpy as np
import os

#Load model YOLO and configuration file 
model_path = os.getenv('MODEL_PATH')
config_path = os.getenv('CONFIG_PATH')

net = cv2.dnn.readNet(model_path, config_path)

names_path = os.getenv('NAMES_PATH')
with open(names_path, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

video_path = os.getenv('VIDEO_PATH')
print(f"Reading video from {video_path}")
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        print("End of video")
        break

    # Prepare the frame for YOLO by resizing and creating a blob
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get YOLO output layers
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Perform forward pass to get detections
    detections = net.forward(output_layers)

    # Initialize lists for storing detection details
    boxes, confidences, class_ids = [], [], []

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Only consider detections with confidence > 0.5
            if confidence > 0.5:
                # YOLO returns normalized coordinates; scale them
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, box_width, box_height) = box.astype("int")

                # Convert to top-left corner coordinates
                x = int(centerX - (box_width / 2))
                y = int(centerY - (box_height / 2))

                boxes.append([x, y, int(box_width), int(box_height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    # Apply non-maxima suppression to remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Loop over the remaining boxes and draw them on the frame
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"

            # Draw bounding box and label if the class is 'person'
            if class_names[class_ids[i]] == "person":
                color = (0, 255, 0)  # Green for person
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame with the detections
    cv2.imshow("YOLO Person Detection", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()