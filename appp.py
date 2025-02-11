import os
import cv2
import torch
import streamlit as st
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog

def setup_cfg(config_path, weights_path):
    """Set up the Detectron2 config."""
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75  # Adjust confidence threshold
    return cfg

def visualize_output(image, outputs, metadata):
    """Visualize predictions with transparent red masks and red outlines."""

    label_classes = metadata.get("thing_classes", ["weed", "paddy"])
    if not label_classes:
        raise ValueError("Metadata missing 'thing_classes'. Check dataset registration.")

    instances = outputs["instances"].to("cpu")
    pred_classes = instances.pred_classes.numpy()
    pred_scores = instances.scores.numpy()
    selected_indices = np.where(np.isin(pred_classes, [0, 1]))[0]  # Filter only weed and paddy

    if len(selected_indices) == 0:
        return image  # No valid predictions

    selected_instances = instances[selected_indices]
    
    # Extract masks
    masks = selected_instances.pred_masks.numpy().astype(np.uint8)

    # Create transparent red overlay
    red_overlay = np.zeros_like(image, dtype=np.uint8)
    red_overlay[:, :, 2] = 150  # Strong red
    red_overlay[:, :, 1] = 10   # Low green
    red_overlay[:, :, 0] = 10   # Low blue

    # Overlay the mask
    overlay = cv2.addWeighted(image, 0.5, red_overlay, 0.9, 0)
    output_image = np.where(np.any(masks, axis=0)[..., None], overlay, image)

    # Draw red outlines around detected masks
    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output_image, contours, -1, (0, 0, 255), 2)  # Red outline

    # Add labels with confidence scores
    for i in selected_indices:
        mask = instances.pred_masks[i].numpy()
        y, x = np.where(mask)
        if len(y) > 0 and len(x) > 0:
            if pred_classes[i] < len(label_classes):
                label = label_classes[pred_classes[i]]
                score = pred_scores[i]
                position = (int(np.mean(x)), int(np.mean(y)))  # Label position
                cv2.putText(
                    output_image,
                    f"{label}: {score:.2f}",
                    position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,  # Font size
                    (255, 255, 255),  # Text color
                    2,  # Thickness
                    cv2.LINE_AA,
                )

    return output_image


def run_inference(cfg, input_image):
    """Run inference on the input image and return the output with masks and detected labels."""
    predictor = DefaultPredictor(cfg)

    # Perform inference
    outputs = predictor(input_image)

    print("Predicted Classes:", outputs["instances"].pred_classes)  # Check detected classes
    print("Confidence Scores:", outputs["instances"].scores)

    # Get metadata, ensuring thing_classes exist
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    thing_classes = getattr(metadata, "thing_classes", ["weed", "Paddy"])  # Default if missing

    # Extract detected labels
    pred_classes = outputs["instances"].pred_classes.cpu().numpy()
    detected_labels = [thing_classes[i] for i in pred_classes if i < len(thing_classes)]
    pred_scores = outputs["instances"].scores.cpu().numpy()

    # Visualize results
    output_image = visualize_output(input_image, outputs, metadata)

    return output_image, detected_labels, pred_scores


def main():
    st.title("Weed Detection and Segmentation")
    
    # Upload image
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read uploaded image
        img_bytes = uploaded_file.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        input_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if input_image is None:
            st.error("Error loading image")
            return

        # Set up config and model paths
        config_path = "/home/prajjwal/Paddy_weed_detection/config.yaml"
        weights_path = "/home/prajjwal/Paddy_weed_detection/output/model_0013999.pth"

        # Validate file paths
        if not os.path.exists(config_path):
            st.error(f"Config file not found: {config_path}")
            return
        if not os.path.exists(weights_path):
            st.error(f"Weights file not found: {weights_path}")
            return

        # Set up Detectron2 config
        cfg = setup_cfg(config_path, weights_path)

        # Run inference
        output_image, detected_labels, pred_scores = run_inference(cfg, input_image)

        # Display results
        st.image(output_image, channels="BGR", caption="Processed Image")

        # Show detected classes and scores
        st.write("### Detected Objects")
        for label, score in zip(detected_labels, pred_scores):
            st.write(f"**{label}** - {score:.2f}")

        # Provide download button
        output_image_bytes = cv2.imencode(".jpg", output_image)[1].tobytes()
        st.download_button("Download Processed Image", data=output_image_bytes, file_name="output_image.jpg", mime="image/jpeg")

if __name__ == "__main__":
    main()
