import streamlit as st
import os
import shutil
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO  # Import YOLO from ultralytics


# Load YOLO Model using Ultralytics
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path, task='detect')  # Load the YOLO model from the provided path
    return model

# Load the Images
@st.cache_data
def load_images(image_folder):
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    return image_paths

# Preprocess Image for YOLO Model (resize and normalize)
def preprocess_image(img):
    img = img.resize((640, 640))  # Resize
    img = np.array(img).astype(np.float32) / 255.0  # Normalize
    img = np.transpose(img, (2, 0, 1))  # Convert (H, W, C) → (C, H, W)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Process YOLO Output to Bounding Boxes (Only Class 0)
def process_yolo_output(results, img_shape, conf_threshold=0.5):
    """ Process YOLO model output and convert to bounding boxes for class 0 only. """
    h, w, _ = img_shape
    detections = []
    
    # Loop through each detection result
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Extract bounding box coordinates (x1, y1, x2, y2)
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy()  # Class IDs
        
        for i, confidence in enumerate(confidences):
            if confidence > conf_threshold and class_ids[i] == 0:  # Only include class 0
                x1, y1, x2, y2 = boxes[i]
                detections.append((int(x1), int(y1), int(x2), int(y2), confidence, int(class_ids[i])))
    
    return detections

# Function to save labels in YOLO format
def save_labels(detections, label_file_path, img_width, img_height):
    with open(label_file_path, "w") as f:
        for x1, y1, x2, y2, confidence, class_id in detections:
            # Calculate normalized coordinates and dimensions
            x_center = (x1 + x2) / 2 / img_width  # Normalize x_center
            y_center = (y1 + y2)/2 / img_height  # Normalize y_center
            width = (x2 - x1) / img_width  # Normalize width
            height = (y2 - y1) / img_height  # Normalize height
            # Write in YOLO format
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# Draw Bounding Boxes on Image
def draw_bounding_boxes(img, detections):
    for x1, y1, x2, y2, confidence, class_id in detections:
        label = f"Class {class_id} {confidence:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

def main():
    # Streamlit UI
    st.title("YOLO ONNX Auto Label App")

    # Input for model path
    model_path = st.text_input("Enter Path to YOLO Model (.onnx or .pt)", "model.onnx")

    # Folder input for images
    image_folder = st.text_input("Enter Path to Image Folder", "images")
    output_folder = st.text_input("Enter Path to Output Folder", "output")

    # Load model
    if model_path and os.path.exists(model_path):
        model = load_model(model_path)  # Load the model using the ultralytics YOLO class
        st.success("Model loaded successfully!")

        # Load images from folder
        if image_folder and os.path.exists(image_folder):
            image_paths = load_images(image_folder)
            if 'image_idx' not in st.session_state:
                st.session_state['image_idx'] = 0

            if image_paths:
                selected_image = image_paths[st.session_state['image_idx']]
                #print(selected_image)

                if selected_image:
                    img = Image.open(selected_image).convert("RGB")
                    img_np = np.array(img)

                    # Preprocess and run inference
                    results = model(selected_image)  # Run inference using Ultralytics YOLO
                    detections = process_yolo_output(results, img_np.shape)  # Process the results

                    # Draw bounding boxes
                    annotated_img = draw_bounding_boxes(img_np, detections)

                    # Display result
                    st.image(annotated_img, caption="Detected Objects")

                    # Save results
                    if output_folder and os.path.exists(output_folder):
                        image_name = os.path.basename(selected_image)
                        image_path = os.path.join(image_folder, image_name)

                        # Create subdirectories for images and labels
                        images_output_folder = os.path.join(output_folder, "images")
                        labels_output_folder = os.path.join(output_folder, "labels")

                        # Ensure these directories exist
                        os.makedirs(images_output_folder, exist_ok=True)
                        os.makedirs(labels_output_folder, exist_ok=True)

                        # Updated paths to save the image and labels
                        output_image_path = os.path.join(images_output_folder, image_name)
                        label_file_path = os.path.join(labels_output_folder, f"{os.path.splitext(image_name)[0]}.txt")

                        # Save Image & Labels
                        _, left_btn, right_btn, _ = st.columns([1, 1, 1, 1])
                        with left_btn:
                            save = st.button("Save Labels")
                        with right_btn:
                            bad = st.button("Bad Result")

                        if save:
                            shutil.move(image_path, output_image_path)
                            #cv2.imwrite(output_image_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)) #writes bbox image
                            save_labels(detections, label_file_path, img.width, img.height)

                            st.success(f"Image saved at {output_image_path}")
                            st.success(f"Labels saved at {label_file_path}")

                            st.session_state['image_idx'] += 1
                            if st.session_state['image_idx'] >= len(image_paths):
                                st.session_state['finished'] = True
                            st.rerun()
                        if bad:
                            # Ensure the for_manual folder exists
                            for_manual_folder = os.path.join(output_folder, "for_manual")
                            os.makedirs(for_manual_folder, exist_ok=True)

                            shutil.move(image_path, os.path.join(for_manual_folder, image_name))
                            st.success(f"Moved {image_name} to 'for_manual' folder")
                            
                            st.session_state['image_idx'] += 1                            
                            if st.session_state['image_idx'] >= len(image_paths):
                                st.session_state['finished'] = True
                            st.rerun()
                    else:
                        st.error("Please enter a valid output folder path.")
            else:
                st.error("No valid images found in the provided folder.")
        else:
            st.error("Please enter a valid image folder path.")


if 'finished' in st.session_state and st.session_state['finished']:
    st.title("Finished labeling!")
else:
    main()
