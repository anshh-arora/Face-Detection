# Face Detection and Quality Check

## Overview

This project involves face detection and quality assessment of images using MediaPipe and OpenCV. The goal is to classify and process images based on the quality and frontal view of detected faces. The images are categorized into different folders based on whether the face is frontal, non-frontal, or the image quality is acceptable.

## Face Detection

Face detection is a computer vision task where the goal is to identify and locate human faces within an image. This is achieved by analyzing the image and detecting key facial landmarks such as the eyes, nose, and mouth. MediaPipe’s Face Mesh is used here to detect these landmarks and classify whether a face is frontal or not.

## Code Workflow

1. **Initialization and Folder Setup**
   - Define folder paths for input images and output categories.
   - Clear existing content in output folders and recreate them.

2. **Load and Process Images**
   - Read each image from the specified folder.
   - Convert the image to RGB format for processing with MediaPipe.

3. **Face Detection**
   - Use MediaPipe’s Face Mesh to detect faces and their landmarks.
   - Check if there are multiple faces in the image. If so, move the image to the "Rejected Quality" folder.

4. **Face Classification**
   - If a single face is detected, check if it is frontal based on landmarks.
   - Crop the face from the image with margins.

5. **Quality Check**
   - Assess the quality of the cropped face image:
     - Sharpness
     - Brightness
     - Darkness
     - Blurriness
   - Save the image based on quality assessment:
     - Frontal faces with good quality are saved in the "Frontal Cropped Faces" folder.
     - Low-quality images or non-frontal faces are saved in the "Rejected Quality" folder.

6. **Final Output**
   - Provide information about saved images based on detection and quality.

## Folder Structure

- `Test_images_data`: Contains the raw images to be processed.
- `Frontal_face_data`: Stores images identified as having frontal faces.
- `Non_Frontal_face_data`: Stores images with non-frontal faces.
- `Rejected_Quality_images`: Stores images that are either low quality or have multiple faces detected.
- `Frontal_Cropped_face_data`: Contains cropped images of frontal faces that passed the quality check.

## Flow Chart

| Step                               | Action                                                                 |
|------------------------------------|------------------------------------------------------------------------|
| **Start**                          | Begin the process.                                                     |
| **Clear and Recreate Folders**     | Clean up and set up output directories.                                 |
| **Load and Process Images**        | Read images from the designated folder.                                 |
| **Detect Faces with MediaPipe**    | Use MediaPipe to find faces and landmarks.                              |
| **Multiple Faces Detected?**       | Check if more than one face is present in the image.                    |
| - Yes                              | Move image to the "Rejected Quality" folder.                            |
| - No                               | Continue to check if the face is frontal.                               |
| **Is Face Frontal?**               | Determine if the detected face is frontal.                              |
| - Yes                              | Save the cropped face image to the "Frontal Cropped Faces" folder.     |
| - No                               | Save the image to the "Non-Frontal Faces" folder.                       |
| **End**                            | Finish the process.                                                     |



## Conclusion

This script automates the process of face detection and quality assessment for a batch of images, categorizing them based on whether they contain frontal or non-frontal faces and the overall image quality. The organized output folders help in easily accessing and reviewing the processed images.

Feel free to modify and expand upon this script according to your needs. If you have any questions or need further assistance, please reach out!
