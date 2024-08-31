import os
import numpy as np
import cv2
import mediapipe as mp
import shutil

# Define file paths (Replace these placeholders with your actual folder paths)
image_folder = r"C:\Users\arora\Desktop\Ansh programs\smartworks\Face-Detection\Test_images_data"
frontal_faces_folder = r"C:\Users\arora\Desktop\Ansh programs\smartworks\Face-Detection\Frontal_Cropped_face_data"
non_frontal_faces_folder = r"C:\Users\arora\Desktop\Ansh programs\smartworks\Face-Detection\Non_Frontal_face_data"
rejected_quality_folder = r"C:\Users\arora\Desktop\Ansh programs\smartworks\Face-Detection\Rejected_Quality_images"
frontal_cropped_faces_folder = r"C:\Users\arora\Desktop\Ansh programs\smartworks\Face-Detection\Frontal_Cropped_face_data"

# Function to clear and recreate folders
def clear_folders():
    for folder in [frontal_faces_folder, non_frontal_faces_folder, rejected_quality_folder, frontal_cropped_faces_folder]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Function to determine if a face is frontal based on landmarks
def is_frontal_face(landmarks):
    nose = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[263]

    eye_to_eye_dist = abs(left_eye.x - right_eye.x)
    nose_to_eye_dist = abs(nose.x - ((left_eye.x + right_eye.x) / 2))

    return nose_to_eye_dist < (eye_to_eye_dist / 5)

# Function to crop the face with margins
def crop_face(image, landmarks, margin=0.3):
    h, w, _ = image.shape
    # Get the bounding box coordinates
    x_coords = [landmark.x * w for landmark in landmarks]
    y_coords = [landmark.y * h for landmark in landmarks]
    
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))
    
    # Calculate margins
    x_margin = int((x_max - x_min) * margin)
    y_margin = int((y_max - y_min) * margin)
    
    # Crop the face with margins
    x_min = max(x_min - x_margin, 0)
    y_min = max(y_min - y_margin, 0)
    x_max = min(x_max + x_margin, w)
    y_max = min(y_max + y_margin, h)
    
    return image[y_min:y_max, x_min:x_max]

# Calculate sharpness using edge detection
def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_count = np.sum(edges > 0)
    return edge_count

# Calculate blurriness
def calculate_blurriness_canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.sum(edges == 0)  # Less edges mean more blur

# Quality check
def quality_check(image, sharpness_threshold=2100, brightness_threshold_low=30.0, brightness_threshold_high=220.0, dark_threshold=20.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = calculate_blurriness_canny(image)
    brightness = np.mean(gray)
    darkness = 255 - brightness
    
    reasons = []
    if blur < sharpness_threshold:
        reasons.append(f"Image is not sharp (sharpness: {blur})")
    if brightness < brightness_threshold_low or brightness > brightness_threshold_high:
        reasons.append(f"Image brightness is not accurate (brightness: {brightness})")
    if darkness < dark_threshold:
        reasons.append(f"Image is dark (darkness: {darkness})")

    if reasons:
        return False, reasons
    return True, [] 

# Clear folders before starting
clear_folders()

# Process each image in the folder
for image_file in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Skipping {image_file}: Unable to read image.")
        continue

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=10, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            if len(results.multi_face_landmarks) > 1:
                # If more than one face is detected, save to rejected folder
                # draw_face_mesh(image, results.multi_face_landmarks[0])
                cv2.putText(image, "More than 1 face detected", (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                save_path = os.path.join(rejected_quality_folder, image_file)
                cv2.imwrite(save_path, image)
                print(f"Saved {image_file} to {save_path} due to multiple faces detected.")
            else:
                face_landmarks = results.multi_face_landmarks[0]
                # Draw face mesh on the original image
                # draw_face_mesh(image, face_landmarks)
                if is_frontal_face(face_landmarks.landmark):
                    # Save the cropped face image with face mesh lines
                    cropped_face = crop_face(image, face_landmarks.landmark)
                    cropped_face_with_mesh = cropped_face.copy()
                    # draw_face_mesh(cropped_face_with_mesh, face_landmarks)
                    
                    # Perform quality check on cropped face
                    is_quality_good, reasons = quality_check(cropped_face_with_mesh)
                    if is_quality_good:
                        cropped_face_path = os.path.join(frontal_cropped_faces_folder, image_file)
                        cv2.imwrite(cropped_face_path, cropped_face_with_mesh)
                        print(f"Saved cropped face with mesh from {image_file} to {cropped_face_path}")
                    else:
                        cv2.putText(cropped_face_with_mesh, "Low Quality", (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                        save_path = os.path.join(rejected_quality_folder, image_file)
                        cv2.imwrite(save_path, cropped_face_with_mesh)
                        print(f"Saved {image_file} to {save_path} due to quality issues: {', '.join(reasons)}")
                else:
                    cv2.putText(image, "Non-Frontal Face Detected", (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                    save_path = os.path.join(non_frontal_faces_folder, image_file)
                    cv2.imwrite(save_path, image)
                    print(f"Saved {image_file} to {save_path}")
        else:
            # Handle cases where no faces are detected or quality is low
            cv2.putText(image, "No Face Detected", (50, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            save_path = os.path.join(rejected_quality_folder, image_file)
            cv2.imwrite(save_path, image)
            print(f"Saved {image_file} to {save_path} due to quality issues or no face detected.")
