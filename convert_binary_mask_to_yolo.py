import cv2
import numpy as np
import matplotlib.pyplot as plt

def convert_to_yolo_format(image_shape, bbox):
    """
    Converts bounding box coordinates to YOLO format.
    
    Parameters:
        image_shape (tuple): The shape of the image as (height, width).
        bbox (tuple): Bounding box in the format (x_min, y_min, x_max, y_max).
    
    Returns:
        list: Bounding box in YOLO format [class_id, x_center, y_center, width, height].
    """
    img_height, img_width = image_shape[:2]
    x_min, y_min, x_max, y_max = bbox
    
    # Calculate YOLO format values
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    return [0, x_center, y_center, width, height]  # class_id is 0

def extract_bounding_boxes(mask_image):
    """
    Extracts bounding boxes from a binary mask image.
    
    Parameters:
        mask_image (numpy.ndarray): Binary mask image (0 and 255 values).
    
    Returns:
        list: List of bounding boxes in (x_min, y_min, x_max, y_max) format.
    """
    # Find contours of the binary mask
    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    
    for contour in contours:
        # Get the bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, x + w, y + h))  # (x_min, y_min, x_max, y_max)
    
    return bounding_boxes

def save_yolo_format(bounding_boxes, image_shape, output_file):
    """
    Saves bounding boxes in YOLO format to a text file.
    
    Parameters:
        bounding_boxes (list): List of bounding boxes in (x_min, y_min, x_max, y_max) format.
        image_shape (tuple): The shape of the image as (height, width).
        output_file (str): Path to the output text file.
    """
    with open(output_file, 'w') as f:
        for bbox in bounding_boxes:
            yolo_bbox = convert_to_yolo_format(image_shape, bbox)
            # Convert to string and write to file
            yolo_line = " ".join(map(str, yolo_bbox))
            f.write(yolo_line + '\n')

def plot_bounding_boxes(mask_image, bounding_boxes):
    """
    Plots the bounding boxes on the binary mask image.
    
    Parameters:
        mask_image (numpy.ndarray): Binary mask image (0 and 255 values).
        bounding_boxes (list): List of bounding boxes in (x_min, y_min, x_max, y_max) format.
    """
    # Create a color image for visualization
    color_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
    
    for bbox in bounding_boxes:
        x_min, y_min, x_max, y_max = bbox
        # Draw rectangle
        cv2.rectangle(color_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    # Plot the image
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    plt.title("Bounding Boxes on Binary Mask")
    plt.axis('off')
    plt.show()

def main(input_mask_path, output_txt_path):
    """
    Main function to process the binary mask image and save bounding boxes in YOLO format.
    
    Parameters:
        input_mask_path (str): Path to the binary mask image.
        output_txt_path (str): Path to the output text file.
    """
    # Read the binary mask image
    mask_image = cv2.imread(input_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_image is None:
        print("Error: Could not read the input mask image.")
        return
    
    # Ensure the mask is binary (values 0 and 255)
    _, binary_mask = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)
    
    # Extract bounding boxes
    bounding_boxes = extract_bounding_boxes(binary_mask)
    
    # Save bounding boxes in YOLO format
    save_yolo_format(bounding_boxes, binary_mask.shape, output_txt_path)
    print(f"Bounding boxes saved in YOLO format to {output_txt_path}")
    
    # Plot the bounding boxes
    plot_bounding_boxes(binary_mask, bounding_boxes)


# Example usage
if __name__ == "__main__":
    input_mask_path = "samples/samp3.jpg"  # Replace with your binary mask image path
    output_txt_path = "samples/samp3_yolo.txt"  # Replace with your desired output file path
    main(input_mask_path, output_txt_path)
