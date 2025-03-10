import cv2
import numpy as np
import argparse
import os

def detect_faces_in_image(image_path):
    """
    Detect faces in a single image file
    
    Args:
        image_path (str): Path to the image file
    """
    # Check if file exists
    if not os.path.isfile(image_path):
        print(f"Error: File {image_path} does not exist")
        return
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load the face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    print(f"Found {len(faces)} faces!")
    
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display the result
    cv2.imshow("Faces found", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_faces_webcam():
   
    # Detect faces in real-time using webcam
    
    # Load the face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Start video capture
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press 'q' to quit")
    
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display the number of faces detected
        cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the resulting frame
        cv2.imshow('Face Detection', frame)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the capture and close windows
    video_capture.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Face Detection using OpenCV')
    parser.add_argument('--image', type=str, help='Path to the image file')
    args = parser.parse_args()
    
    if args.image:
        detect_faces_in_image(args.image)
    else:
        detect_faces_webcam()

if __name__ == "__main__":
    main() 