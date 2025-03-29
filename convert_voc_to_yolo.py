import os
import xml.etree.ElementTree as ET

# A dictionary mapping class names to YOLO class IDs
class_map = {
    'dog': 0,
    'cat': 1,
    # Add more classes as needed
}

def voc_to_yolo(voc_xml_path, output_txt_path):
    """
    Converts a VOC XML annotation file to YOLO format and saves it as a text file.
    """
    # Parse the XML file
    tree = ET.parse(voc_xml_path)
    root = tree.getroot()

    # Get image size
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)

    # Open the output text file for writing
    with open(output_txt_path, 'w') as f:
        # Loop through all objects in the XML
        for obj in root.findall('object'):
            # Get the object class name
            class_name = obj.find('name').text

            # Check if the class exists in the class map
            if class_name not in class_map:
                print(f"Class '{class_name}' not found in class_map. Skipping...")
                continue

            # Get the class ID
            class_id = class_map[class_name]

            # Get bounding box coordinates
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)

            # Normalize the bounding box coordinates
            x_center = (xmin + xmax) / 2 / width
            y_center = (ymin + ymax) / 2 / height
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height

            # Write the YOLO format label to the text file
            f.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")

    print(f"Converted VOC to YOLO format and saved to {output_txt_path}")


# Example usage
voc_xml_path = "samples/samp1_voc.xml"
output_txt_path = "samples/samp1_yolo.txt"
voc_to_yolo(voc_xml_path, output_txt_path)
