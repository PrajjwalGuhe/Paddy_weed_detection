import os
import cv2
import torch
import streamlit as st
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog
import numpy as np

def setup_cfg(config_path, weights_path):
    """Set up the Detectron2 config."""
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # Set threshold for this run
    return cfg

def visualize_output(image, outputs, metadata):
    """Visualize predictions with reduced opacity red masks and add labels with confidence scores."""
    # Ensure metadata has thing_classes
    label_classes = metadata.get("thing_classes", ["weed", "Paddy"])
    if not label_classes:
        raise ValueError("Metadata does not contain 'thing_classes'. Check dataset registration.")

    # Get instances and filter predictions for classes 0 and 3
    instances = outputs["instances"]
    pred_classes = instances.pred_classes.cpu().numpy()
    pred_scores = instances.scores.cpu().numpy()
    selected_indices = np.where(np.isin(pred_classes, [0,1]))[0]
    selected_instances = instances[selected_indices]

    # Extract masks and combine them
    masks = selected_instances.pred_masks.cpu().numpy().astype(np.uint8)
    combined_mask = np.sum(masks, axis=0).clip(0, 1)  # Binary mask

    # Create a blood red overlay with reduced opacity
    blood_red_overlay = np.zeros_like(image, dtype=np.uint8)
    blood_red_overlay[:, :, 2] = 150  # Increase the red channel intensity for a darker red
    blood_red_overlay[:, :, 1] = 10  # Keep the green channel low
    blood_red_overlay[:, :, 0] = 10   # Keep the blue channel low

    # Blend the original image with the blood-red overlay
    overlay = cv2.addWeighted(image, 0.5, blood_red_overlay, 0.9, 0)

    # Apply the red overlay where the mask exists
    output_image = np.where(combined_mask[..., None], overlay, image)

    # Add labels and confidence scores
    for i in selected_indices:
        mask = instances.pred_masks[i].cpu().numpy()
        y, x = np.where(mask)
        if len(y) > 0 and len(x) > 0:
            if pred_classes[i] < len(label_classes):  # Ensure index is valid
                label = label_classes[pred_classes[i]]
                score = pred_scores[i]
                position = (int(np.mean(x)), int(np.mean(y)))  # Position for the label
                cv2.putText(
                    output_image,
                    f"{label}: {score:.2f}",
                    position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # Font size
                    (255, 255, 255),  # Text color
                    1,  # Thickness
                    cv2.LINE_AA,
                )

    return output_image


def run_inference(cfg, input_image):
    """Run inference on the input image and return the output with masks."""
    predictor = DefaultPredictor(cfg)

    # Perform inference
    outputs = predictor(input_image)

    print("Predicted Classes:", outputs["instances"].pred_classes)  # Check detected classes
    print("Confidence Scores:", outputs["instances"].scores) 

    # Visualize the results
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    output_image = visualize_output(input_image, outputs, metadata)

    return output_image

def main():
    st.title("Weed Detection and Segmentation")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image
        img_bytes = uploaded_file.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        input_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if input_image is None:
            st.error("Error loading image")
            return

        # Set up config and model
        config_path = "/home/prajjwal/Paddy_weed_detection/config.yaml"
        weights_path = "/home/prajjwal/Paddy_weed_detection/output/model_0013999.pth"
        if not os.path.exists(config_path):
            st.error(f"Config file not found: {config_path}")
            return
        if not os.path.exists(weights_path):
            st.error(f"Weights file not found: {weights_path}")
            return

        # Setup configuration
        cfg = setup_cfg(config_path, weights_path)

        # Run inference and get the output image
        output_image = run_inference(cfg, input_image)

        # Display the output image
        st.image(output_image, channels="BGR", caption="Processed Image")

        # Optionally, provide download button
        output_image_bytes = cv2.imencode(".jpg", output_image)[1].tobytes()
        st.download_button("Download Processed Image", data=output_image_bytes, file_name="output_image.jpg", mime="image/jpeg")

if __name__ == "__main__":
    main()
