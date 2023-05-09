import os
import cv2

# Set the input and output paths
input_path = "people.jpg"
output_path = "/cropped_faces/"

# Load the image and convert it to grayscale
img = cv2.imread(input_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create the face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Loop over all detected faces
if len(faces) > 0:
    for i, (x, y, w, h) in enumerate(faces):
        # Draw a rectangle around the face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        
        # Crop the face
        face = img[y:y + h, x:x + w]
        
        # Save the cropped face to the output folder
        filename = os.path.join(output_path, f"face{i}.jpg")
        cv2.imwrite(filename, face)
        print(f"{filename} is saved")

# Display the image with detected faces
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
