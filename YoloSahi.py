# pip install -U torch sahi ultralytics

# arrange an instance segmentation model for test
from sahi.utils.ultralytics import (
    download_yolo11n_model, download_yolo11n_seg_model,
    # download_yolov8n_model, download_yolov8n_seg_model
)

from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict

import cv2
import matplotlib.pyplot as plt


def load_model(model_path, model_type = 'yolo11', confidence_threshold=0.3, device='cuda:0'):
    """Load the YOLOv11 model for SAHI inference."""
    # Load the model
    detection_model = AutoDetectionModel.from_pretrained(
        model_type=model_type,
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        device=device
    )
    return detection_model


def perform_sliced_prediction(image_path, model, slice_height=256, slice_width=256, overlap_height_ratio=0.2,
                              overlap_width_ratio=0.2):
    """Perform small object detection using SAHI slicing."""
    image = read_image(image_path)
    result = get_sliced_prediction(
        image,
        model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )
    return result


def visualize_results(image_path, result):
    """Visualize detection results."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for obj in result.object_prediction_list:
        bbox = obj.bbox.to_xyxy()
        x1, y1, x2, y2 = map(int, bbox)
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', linewidth=2, fill=False))
        plt.text(x1, y1 - 5, obj.category.name, color='red', fontsize=12, weight='bold')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":

    model_path = "models/yolov11n.pt"
    download_yolo11n_model(model_path)
    # yolov8n_model_path = "models/yolov8n.pt"
    # download_yolov8n_model(yolov8n_model_path)

    # download test images into demo_data folder
    download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg',
                      'demo_data/small-vehicles1.jpeg')
    download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/terrain2.png',
                      'demo_data/terrain2.png')
    image_path = "demo_data/small-vehicles1.jpeg"  # Path to the test image

    # Load the model
    detection_model = load_model(model_path)
    # # Perform prediction
    # result = get_prediction(image_path, detection_model)
    # Perform sliced prediction
    result = perform_sliced_prediction(image_path, detection_model)

    # Visualize results
    visualize_results(image_path, result)

    # result.to_coco_predictions(image_id=1)[:3]